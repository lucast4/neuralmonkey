import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from neuralmonkey.neuralplots.population import plotNeurHeat, plotNeurTimecourse
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type

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
            self.Chans = list(range(self.X.shape[0]))
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
        """ z-score across trials and time bins, separately for each chan.
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
        if dim in ["site", "sites", "chans", "trials"]:
            # 1) Map the values to indices
            # dim, dim_str = self.help_get_dimensions(dim)
            indices = self.index_find_these_values(dim, values)
        elif dim=="times":
            # values are [t1, t2]
            if not len(values)==2:
                print(values)
                assert False, "why?"
            assert values[1]>values[0]
            indices = self.index_find_this_time_window(values, time_keep_only_within_window=time_keep_only_within_window)
            # indices = self.index_find_these_values(dim, values)
            assert len(indices)==2
        else:
            print(dim)
            assert False, "not correct dim"

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

        Makes copies of Xlabels dataframes
        RETURNS:
        - PopAnal object, , copy
        - OR frmat array (chans, trials, times) if return_only_X==True
        """

        # convert to int
        dim, dim_str = self.help_get_dimensions(dim)

        if dim_str=="times":
            if len(inds)==1:
                # Assume you want to just get a single bin
                inds = [inds[0], inds[0]]
                
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
                from pythonlib.tools.pandastools import _check_index_reseted
                _check_index_reseted(self.Xlabels["trials"])
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

        if reset_trial_indices and dim_str=="trials":
            # assert dim_str=="trials", "this doesnt make sense otherwise. mistake?"
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

        if len(chans)==0:
            # Then return empty
            assert False

        # convert from channel labels to row indices
        if chan_inputed_row_index:
            inds = chans
            # del chans
        else:
            inds = [self.index_find_this_chan(ch) for ch in chans]
            # del chans

        if len(self.Chans)<max(inds)+1:
            # Then not enough chans
            print(version)
            print(self.X.shape)
            print(inds)
            print(self.Chans)
            print(chans)
            print(chan_inputed_row_index)
            assert False, "you need to input only chans that exist"

        # Slice
        try:
            X = self.extract_activity_copy(version=version)
            X = X[inds, :, :]
        except Exception as err:
            print(version)
            print(self.X.shape)
            print(X.shape)
            print(inds)
            print(self.Chans)
            print(chans)
            print(chan_inputed_row_index)
            raise err

        if return_as_popanal:
            chans_actual = [self.Chans[i] for i in inds]
            PA = PopAnal(X, times=self.Times, trials=self.Trials, chans=chans_actual)
            return PA
        else:
            return X
    
    def replace_X(self, X, times, chans):
        """ Replace X-- modifying self
        """
        assert X.shape[1]==self.X.shape[1], "assumes trials are matches, to get labels"
        # trials = self.Trials

        if chans is None and (X.shape[0]==self.X.shape[0]):
            chans = self.Chans

        if times is None and (X.shape[2]==self.X.shape[2]):
            times = self.Times

        self.X = X
        self.Chans = chans
        self.Times = times

        # Remove these, just in case they are inconssitent with new inputs
        # pa.Xlabels["trials"] = self.Xlabels["trials"].copy()
        self.Xlabels["chans"] = pd.DataFrame()
        self.Xlabels["times"] = pd.DataFrame()

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

    def mean_over_trials(self, version="raw", add_to_flank=0.001):
        """ Return X, but mean over trials,
        out shape (nchans, 1, time)
        """
        assert False, "OBSOLETE. use agg_wrapper, it retains Xlabels"
        X = self.extract_activity_copy(version=version)
        return np.mean(X, axis=1, keepdims=True)

    def _agg_by_time_windows_binned_get_windows(self, DUR, SLIDE, min_time=None, max_time=None):
        """
        Helper to get the windows that you can then use for biniing.
        PARAMS:
        - min_time, max_time, sec, if not None, keeps only times within this window.
        - Returns (nwinds, 2) array
        """

        if SLIDE is None:
            SLIDE = DUR

        times = self.Times
        if min_time is not None:
            times = times[times>=min_time]
        if max_time is not None:
            times = times[times<=max_time]
        
        # MAke new times iwndows
        PRE = times[0]
        POST = times[-1]
        times1 = np.arange(PRE, POST-DUR/2, SLIDE) # each window, at least half of it inlcudes data
        times2 = times1 + DUR
        time_windows = np.stack([times1, times2], axis=1)
        assert np.isclose(time_windows[0,0], PRE)
        # assert time_windows[-1,1]>=post
        assert time_windows[-1,0]<POST

        # Add to flank, to abvoid numerical miss
        time_windows[:, 0] = time_windows[:, 0]-0.001
        time_windows[:, 1] = time_windows[:, 1]+0.001

        return time_windows

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

        if DUR is None:
            return self.copy()
        
        # Only contineu if DUR is larger than the largest period between adjacent samples.
        max_period = np.max(np.diff(self.Times))

        if DUR >= (1.001 * max_period):

            if False:
                # 1. Original -- SLOW!!
                time_windows = self._agg_by_time_windows_binned_get_windows(DUR, SLIDE)

                # if SLIDE is None:
                #     SLIDE = DUR

                # # MAke new times iwndows
                # PRE = self.Times[0]
                # POST = self.Times[-1]
                # if False:
                #     # Failed soemtimes, if dur > amount of data. then time_wind would be []
                #     # n = (POST-PRE)/DUR
                #     times1 = np.arange(PRE, POST-DUR, SLIDE)
                #     times2 = times1+DUR
                #     time_windows = np.stack([times1, times2], axis=1)
                # else:
                #     times1 = np.arange(PRE, POST-DUR/2, SLIDE) # each window, at least half of it inlcudes data
                #     times2 = times1 + DUR
                #     time_windows = np.stack([times1, times2], axis=1)
                #     assert np.isclose(time_windows[0,0], PRE)
                #     # assert time_windows[-1,1]>=post
                #     assert time_windows[-1,0]<POST
                

                # print("===========")
                # print(DUR, SLIDE)
                # print(time_windows)
                # print(self.Times)
                # assert False

                X_binned, times_binned = self.agg_by_time_windows(time_windows)
            elif False:
                # 2. New, ChatGPT, works, but might fail in some cases with weird bins
                # THis should be faster, esp for large data
                # Tested and checked that it works.
                # Assuming X is your neural data with shape (neurons, trials, times)
                # and T is the time vector with units of seconds

                T = self.Times
                X = self.X

                # Calculate the sampling interval
                dt = T[1] - T[0]  # Assuming uniform sampling

                # Calculate the number of points per bin and step
                bin_size_points = int(DUR / dt)
                step_size_points = int(SLIDE / dt)

                # Initialize lists to store the binned data and time stamps
                X_binned = []
                T_binned = []

                total_time_points = X.shape[2]

                # Sliding window loop
                for start in range(0, total_time_points - bin_size_points + 1, step_size_points):
                    end = start + bin_size_points  # End index for the current bin
                    
                    # Extract data for the current bin
                    X_window = X[:, :, start:end]  # Shape: (neurons, trials, bin_size_points)
                    
                    # Compute the mean across the time dimension (axis=2)
                    X_mean = X_window.mean(axis=2)  # Shape: (neurons, trials)
                    
                    # Append the mean data to the list
                    X_binned.append(X_mean)
                    
                    # Compute and append the center time of the bin
                    bin_center_time = T[start] + (DUR / 2)
                    T_binned.append(bin_center_time)

                # Convert lists to arrays
                X_binned = np.stack(X_binned, axis=2)  # Shape: (neurons, trials, number of bins)
                times_binned = np.array(T_binned)          # Shape: (number of bins,)
            else:
                # 3. The best method, is fast. Gets identical windows to method 1, but muhc fastrs.
                time_windows = self._agg_by_time_windows_binned_get_windows(DUR, SLIDE)

                T = np.array(self.Times)
                X = self.X

                X_binned = []
                T_binned = []
                for i in range(time_windows.shape[0]):
                    twind = time_windows[i, :] # (t1, t2)
                    # print(twind)

                    x = self.X[:, :, (T>=twind[0]) & (T<=twind[1])]
                    X_mean = np.mean(x, axis=2) # (chans, trials)

                    # Append the mean data to the list
                    X_binned.append(X_mean)
                    
                    # Compute and append the center time of the bin
                    bin_center_time = twind[0] + DUR/2
                    T_binned.append(bin_center_time)

                # Convert lists to arrays
                X_binned = np.stack(X_binned, axis=2)  # Shape: (neurons, trials, number of bins)
                times_binned = np.array(T_binned)          # Shape: (number of bins,)

            PA = PopAnal(X_binned, times=times_binned, chans=self.Chans,
                        trials = self.Trials)
            PA.Xlabels["trials"] = self.Xlabels["trials"].copy()
        else:
            PA = self.copy()

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

    def slice_extract_with_levels_of_conjunction_vars_as_dictpa(self, var, vars_others,
                                                      prune_min_n_trials=5, prune_min_n_levs=2,
                                                      plot_counts_heatmap_savepath=None):
        """
        See slice_extract_with_levels_of_conjunction_vars, here does same, but returns as dict, 
        grp:pa
        """

        _, _, dict_dfthis = self.slice_extract_with_levels_of_conjunction_vars(var, vars_others, 
                                                                               prune_min_n_trials=prune_min_n_trials, 
                                                                               prune_min_n_levs=prune_min_n_levs,
                                                                               plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
        dict_pa = {}
        for grp, _df in dict_dfthis.items():
            # print(grp, _df["_index"].tolist())
            dict_pa[grp] = self.slice_by_dim_indices_wrapper("trials", _df["_index"].tolist(), True)
            
        return dict_pa
        
    
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
        # from pythonlib.tools.pandastools import _check_index_reseted
        # _check_index_reseted(dflab)
        dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var, vars_others,
                                                                 n_min_across_all_levs_var=prune_min_n_trials,
                                                                 lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                 prune_levels_with_low_n=True,
                                                                 ignore_values_called_ignore=True,
                                                                 plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
        # print(len(dflab), len(dfout))
        # print(dflab.index)
        # print(dfout.index)

        if len(dfout)>0:
            # Only keep the indices in dfout
            pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)
            return pa, dfout, dict_dfthis
        else:
            return None, None, None

    def slice_prune_dflab_and_vars_others(self, var_effect, vars_others, n_min_per_lev, 
                                          lenient_allow_data_if_has_n_levels, fail_if_prune_all=True):
        """
        Return pruned copy of self, and pruned vars_others, such that data exists, i.e,, that at least <lenient_allow_data_if_has_n_levels>
        number levels of <var_effect> for some levesl of <varsS_others>, with at least <n_min_per_lev> trials.
        
        Will allow to pass if at least 1 datapt.

        Iteratively removes item from end of vars_others.

        """
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        n = 0
        while n==0:
            assert len(vars_others)>0, "all pruned...?"
            dflab = self.Xlabels["trials"]

            if False:
                print("Pruning ... ", var_effect, " -- ", vars_others, " -- ", n_min_per_lev, " -- ", lenient_allow_data_if_has_n_levels)

            dfout, _ = extract_with_levels_of_conjunction_vars_helper(dflab, var=var_effect, vars_others=vars_others, 
                                                    n_min_per_lev=n_min_per_lev, 
                                                    lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels)        
            n = len(dfout)
            if n==0:
                vars_others = vars_others[:-1]
                print("vars_others is now: ", vars_others)
        
        inds = dfout["_index"].tolist()
        if len(inds)==0 and fail_if_prune_all:
            assert False
        pa = self.slice_by_dim_indices_wrapper("trials", inds)

        return pa, vars_others

    def slice_extract_with_levels_of_var_good_prune(self, grp_vars, n_min_per_var):
        """
        Preprocess, pruning to keep onl levels of grouping var which have at least
        <n_min_per_var> many trials
        RETURNS:
        - pa, a copy
        """
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good
        dflab = self.Xlabels["trials"]
        _, inds_keep = extract_with_levels_of_var_good(dflab, grp_vars, n_min_per_var)
        pa = self.slice_by_dim_indices_wrapper("trials", inds_keep, reset_trial_indices=True)
        print("Pruned: ", self.X.shape, " --> ", pa.X.shape)
        return pa

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

                if len(pa.Xlabels["trials"])>0:
                    inds = pa.Xlabels["trials"][pa.Xlabels["trials"][_var].isin(_levs)].index.tolist()
                    print(f"pa.slice_by_labels_filtdict, using var={_var}, n before filt: {pa.X.shape}")
                    pa = pa.slice_by_dim_indices_wrapper("trials", inds)
                    print(f"pa.slice_by_labels_filtdict, using var={_var}, n after filt: {pa.X.shape}")
                    # pa.Xlabels["trials"] = pa.Xlabels["trials"][pa.Xlabels["trials"][_var].isin(_levs)].reset_index(drop=True)
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

    def sort_trials_by_trialcode(self):
        """
        return copy of self, with trials sorted by incresaeing trial (using trialcode)
        """
        from pythonlib.tools.stringtools import trialcode_to_scalar

        dflab = self.Xlabels["trials"]

        # First, get trialcodes into sortable scalrs
        trialcode_scalars = [trialcode_to_scalar(tc) for tc in dflab["trialcode"]]
        dflab["trialcode_scal"] = trialcode_scalars

        sort_indices = dflab["trialcode_scal"].argsort()

        pa = self.slice_by_dim_indices_wrapper("trials", sort_indices, reset_trial_indices=True)
        
        return pa

    def sort_chans_by_fr_in_window(self, twind):
        """
        REturn PA (copy) that has channels sorted in order of incresaing firing rates,
        within time window twind (2-tuple).
        RETURNS:
        - copy of PA, with chans ordered from low FR to high FR
        """
        
        pathis = self.slice_by_dim_values_wrapper("times", twind).agg_wrapper("times").agg_wrapper("trials")
        sortinds_chan = np.argsort(pathis.X[:, 0,0]).tolist()
        PA = self.slice_by_dim_indices_wrapper("chans", sortinds_chan)

        return PA, sortinds_chan

    def sort_chans_by_modulation_over_time(self, PLOT=False):
        """
        Return PA copy that has chans sorted based on moudlation of fr over time.
        Modulation over time is r2, anova, vs time, which is a good metric for how 
        strongly this chan is modulated as a SNR metric.
        RETURNS:
        - copy of PA, with chans ordered from most to least modulated
        """
        from neuralmonkey.metrics.scalar import _calc_modulation_by_frsm_event_aligned_time
        
        # For each chan, compute modulation
        res = []
        for i, chan in enumerate(self.Chans):
            frmat = self.X[i, :, :]
            r2 = _calc_modulation_by_frsm_event_aligned_time(frmat)
            res.append({
                "r2":r2,
                "chan":chan,
                "indchan":i
            })
        df = pd.DataFrame(res)

        if PLOT:
            import seaborn as sns
            from pythonlib.tools.snstools import rotateLabel
            fig = sns.catplot(data=df, x="chan", y="r2", aspect=2.5, kind="bar")
            rotateLabel(fig)
        
        # Sort
        sortinds_chan = df.sort_values("r2", ascending=False)["indchan"].tolist()
        PA = self.slice_by_dim_indices_wrapper("chans", sortinds_chan)

        return PA, sortinds_chan

    def norm_rel_all_timepoints(self, method="zscore"):
        """
        Normalize each channel against stats from its entire data (i.e., trials x times)
        RETURNS:
        - copy of PA, with modified X
        """

        if method=="zscore":
            self.zscoreFrNotDataframe()
            pa = self.copy()
            pa.X = self.Xz
            self.Xz = None
        else:
            print(method)
            assert False

        return pa

    def norm_rel_base_window(self, twind_base, method="zscore", return_stats=False):
        """
        Normalize actrivity by subtractigin mean acitivty taken from a baseline
        time window (and then averaged over time). 
        """

        # Get mean and std for each chan, using a baseline twind
        pa_base = self.slice_by_dim_values_wrapper("times", twind_base)
        x = pa_base.dataextract_reshape("chans_x_trialstimes")
        xmean = np.mean(x, axis=1)[:, None, None]
        xstd = np.std(x, axis=1)[:, None, None]

        # print(self.X.shape)
        # print(pa_base.X.shape)
        # print(xmean.shape)
        # print(xstd.shape)
        # if method=="zscore":
        #     # pa_norm.X = (pa_norm.X - xmean)/xstd
        # elif method=="subtract":
        # else:
        #     assert False

        pa_norm = self.norm_rel_base_apply(xmean, xstd, method)

        if return_stats:
            return pa_norm, xmean, xstd
        else:
            return pa_norm

    def norm_rel_base_apply(self, xmean, xstd, method="zscore"):
        """
        Apply this prcomputed mean and std
        PARAMS:
        - xmean, (nchans, 1, 1) or (nchans,)
        """

        assert (xmean.shape == (self.X.shape[0], 1, 1)) or (xmean.shape == (self.X.shape[0],))
        assert xstd.shape == (self.X.shape[0], 1, 1) or (xstd.shape == (self.X.shape[0],))
        
        pa_norm = self.copy()
        if method=="zscore":
            pa_norm.X = (pa_norm.X - xmean)/xstd
        elif method=="subtract":
            pa_norm.X = pa_norm.X - xmean
        else:
            print(method)
            assert False

        return pa_norm


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
        across all trials within that (chan, timepoint) --> resulting will have
        mean fr of 0 at each (chan, timepoint).
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

        if dim_variable_grp is None:
            # Then just subtract across all trials (ie., all trials constitue one group)
            return self.norm_subtract_mean_each_chan()

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
    
    def split_stratified_constrained_grp_var(self, nsplits, label_grp_vars, fraction_constrained_set=0.5, n_constrained=2, 
                                             list_labels_need_n=None, min_frac_datapts_unconstrained=None, 
                                             min_n_datapts_unconstrained=1, plot_train_test_counts=False, plot_indices=False,
                                             plot_all_folds=False):
        """
        [Good] Split data (trials) in stratitied manner by label, with helping to make sure not have not enough trials in output
        (constraints).

        PARAMS:
        - nsplits, each time does newly with shuffle (with replacment).
        - fraction_constrained_set = 0.75 # Take most for the euclidian distance (less for dpca)
        - n_constrained = 2 # Need at least 2 for pairwise comparisons
        - min_frac_datapts_unconstrained = None
        - min_n_datapts_unconstrained, n trials for unconstrinaed. e.g., if using this for dpca, then could be 
        len(dflab[dpca_var].unique()) # 
        - list_labels_need_n, e..g, [('arcdeep-4-4-0', (-1, 1), 'rig3_3x3_big')]
        """
        from pythonlib.tools.statstools import split_stratified_constrained, split_stratified_constrained_multiple
        from pythonlib.tools.pandastools import _check_index_reseted, grouping_plot_n_samples_conjunction_heatmap_helper

        ### Extract labels
        dflab = self.Xlabels["trials"]
        _check_index_reseted(dflab)
        y = [tuple(x) for x in dflab.loc[:, label_grp_vars].values.tolist()]

        ### Run
        # unconstrained_indices, constrained_indices, _, _ = split_stratified_constrained(y, fraction_constrained_set, 
        #                                                         n_constrained, list_labels_need_n=list_labels_need_n, 
        #                                                         min_frac_datapts_unconstrained=min_frac_datapts_unconstrained,  
        #                                                         min_n_datapts_unconstrained=min_n_datapts_unconstrained, PRINT=False, PLOT=plot_indices)
        try:
            folds = split_stratified_constrained_multiple(y, nsplits, fraction_constrained_set, 
                                                                    n_constrained, list_labels_need_n=list_labels_need_n, 
                                                                    min_frac_datapts_unconstrained=min_frac_datapts_unconstrained,  
                                                                    min_n_datapts_unconstrained=min_n_datapts_unconstrained, 
                                                                    PRINT=False, PLOT=plot_indices)
        except Exception as err:
            fig = grouping_plot_n_samples_conjunction_heatmap_helper(self.Xlabels["trials"], label_grp_vars)
            savefig(fig, "/tmp/counts_tmp.pdf")
            print("check: ", "/tmp/counts_tmp.pdf")
            raise err

        ### Check things
        # Plot coutns (sanity check)
        if plot_train_test_counts:
            # Just plot the first fold

            unconstrained_indices, constrained_indices = folds[0]
            paredu_train = self.slice_by_dim_indices_wrapper("trials", unconstrained_indices)
            paredu_test = self.slice_by_dim_indices_wrapper("trials", constrained_indices)
            grouping_plot_n_samples_conjunction_heatmap_helper(dflab, label_grp_vars)
            fig_unc = grouping_plot_n_samples_conjunction_heatmap_helper(paredu_train.Xlabels["trials"], label_grp_vars)
            fig_con = grouping_plot_n_samples_conjunction_heatmap_helper(paredu_test.Xlabels["trials"], label_grp_vars)

            if plot_all_folds:
                for unconstrained_indices, constrained_indices in folds:
                    paredu_train = self.slice_by_dim_indices_wrapper("trials", unconstrained_indices)
                    paredu_test = self.slice_by_dim_indices_wrapper("trials", constrained_indices)
                    grouping_plot_n_samples_conjunction_heatmap_helper(paredu_train.Xlabels["trials"], label_grp_vars)
                    grouping_plot_n_samples_conjunction_heatmap_helper(paredu_test.Xlabels["trials"], label_grp_vars)
        else:
            fig_unc, fig_con = None, None

        return folds, fig_unc, fig_con

    def split_balanced_stratified_kfold_subsample_level_of_var(self, label_grp_vars, var_exclude=None, levels_exclude_from_splitting=None,
                                        n_splits="auto", do_balancing_of_train_inds=True, plot_train_test_counts=False, plot_indices=False,
                                        shuffle=False):
        """
        Split into n_splits train-test samples, ensuring that trains are balanced across lewvels
        of <label_grp_vars>. 

        Additiaonlly, do this only for specific levels of var.

        e.g., var_exclude=="var", and levels_exclude==[A, B], then does train-test split for each level that is not A, B,..
        and for each fold, includes all of the indices for A an B> This is useufl if you want to do train-test for everything that
        is not A, B. Returns indices into the oriinal Dflab (self)

        NOTE: guaranteed that each index will contribute to test once and only once.

        NOTE: to run this without excluding anything, set levels_exclude_from_splitting=[]
        PARAMS:
        - do_balancing_of_train_inds, bool.
        - label_grp_vars = ["idx_morph_temp", "seqc_0_loc"]
        """
        assert False, "note that this forces lower nsplits if any level has ndata less than nsplits. Instead, use split_stratified_constrained_grp_var (altbough that samples with replacement)"

        # Train-test split
        from pythonlib.tools.statstools import balanced_stratified_kfold
        from pythonlib.tools.pandastools import append_col_with_grp_index, _check_index_reseted

        dflab = self.Xlabels["trials"]
        _check_index_reseted(dflab)
        dflab = append_col_with_grp_index(dflab, label_grp_vars, "tmp")

        if var_exclude is None:
            dflab_inner = dflab
            inds_in_dflab_inner = dflab_inner.index
            inds_in_dflab_base = []
        else:
            dflab_inner = dflab[~dflab[var_exclude].isin(levels_exclude_from_splitting)] # get the non-base inds
            dflab_base = dflab[dflab[var_exclude].isin(levels_exclude_from_splitting)] # get the non-base inds
            inds_in_dflab_inner = dflab_inner.index
            inds_in_dflab_base = dflab_base.index

        y = dflab_inner["tmp"].values.astype(str)

        folds = balanced_stratified_kfold(None, y,  n_splits=n_splits, 
                                          do_balancing_of_train_inds=do_balancing_of_train_inds,
                                          shuffle=shuffle)
        print(n_splits)
        print(len(folds))
        assert False
        
        # map the indices here back to indices in original dflab
        folds_dflab = []
        for train_inds, test_inds in folds:
            train_inds_dflab = inds_in_dflab_inner[train_inds]
            test_inds_dflab = inds_in_dflab_inner[test_inds]

            # also append the base inds, always, for both training and testing.
            train_inds_dflab = np.concatenate([train_inds_dflab, inds_in_dflab_base])
            test_inds_dflab = np.concatenate([test_inds_dflab, inds_in_dflab_base])

            folds_dflab.append([train_inds_dflab, test_inds_dflab])

        # Plot coutns (sanity check)
        if plot_train_test_counts:
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            for train_inds, test_inds in folds_dflab:
                # train_inds, test_inds = folds_dflab[0]
                paredu_train = self.slice_by_dim_indices_wrapper("trials", train_inds)
                paredu_test = self.slice_by_dim_indices_wrapper("trials", test_inds)

                grouping_plot_n_samples_conjunction_heatmap(paredu_train.Xlabels["trials"], "idx_morph_temp", "seqc_0_loc");
                grouping_plot_n_samples_conjunction_heatmap(paredu_test.Xlabels["trials"], "idx_morph_temp", "seqc_0_loc");

        # Plot indices
        if plot_indices:
            fig, ax = plt.subplots(figsize=(15,5))
            for i, (trains, tests) in enumerate(folds_dflab):
                ax.plot(trains, i*np.ones(len(trains)), ".k")
                ax.plot(tests, i*np.ones(len(tests)), "xr")

        return folds_dflab
    

    def split_balanced_stratified_kfold(self, label_grp_vars, n_splits="auto", do_balancing_of_train_inds=True):
        """
        Split into n_splits train-test samples, ensuring that trains are balanced across lewvels
        of <label_grp_vars>. 
        PARAMS:
        - do_balancing_of_train_inds, bool.
        - label_grp_vars = ["idx_morph_temp", "seqc_0_loc"]
        """
        assert False, "this is identical to split_balanced_stratified_kfold_subsample_level_of_var, with levels_exclude_from_splitting=[]. Merge them"
        # Train-test split
        from pythonlib.tools.statstools import balanced_stratified_kfold
        from pythonlib.tools.pandastools import append_col_with_grp_index

        dflab = self.Xlabels["trials"]
        dflab = append_col_with_grp_index(dflab, label_grp_vars, "tmp")

        y = dflab["tmp"].values.astype(str)
        folds = balanced_stratified_kfold(None, y,  n_splits=n_splits, do_balancing_of_train_inds=do_balancing_of_train_inds)
        return folds
    
    def split_sample_stratified_by_label(self, label_grp_vars, test_size=0.5, PRINT=False):
        """
        REturn two evenly slit PA, using up all the trials in self, and mainting the same proportion of classes of
        conj-var label_grp_vars (i.e,, stratified).
        """
        from pythonlib.tools.statstools import balanced_stratified_resample_kfold
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from pythonlib.tools.statstools import stratified_resample_split_kfold

        # COnvert to conjunctive label as strings.
        # assert "tmp" not in self.Xlabels["trials"].columns
        self.Xlabels["trials"] = append_col_with_grp_index(self.Xlabels["trials"], label_grp_vars, "tmp")
        labels = self.Xlabels["trials"]["tmp"].tolist()
        # labels = dflab["tmp"].tolist()

        # Get split indices for the two groups.
        from pythonlib.tools.listtools import tabulate_list
        outdict = tabulate_list(labels)
        if PRINT:
            print("Labels, split:")
            for k, v in outdict.items():
                print(k, " -- ", v)
        split_inds = stratified_resample_split_kfold(labels, 1, test_size=test_size, PRINT=PRINT)

        # Generate new pa
        train_index, test_index = split_inds[0]
        pa1 = self.slice_by_dim_indices_wrapper("trials", train_index.tolist())
        pa2 = self.slice_by_dim_indices_wrapper("trials", test_index.tolist())

        if PRINT:
            print(" --- pa1:")
            display(pa1.Xlabels["trials"]["tmp"].value_counts())
            print(" --- pa2:")
            display(pa2.Xlabels["trials"]["tmp"].value_counts())

        self.Xlabels["trials"] = self.Xlabels["trials"].drop("tmp", axis=1)
        pa1.Xlabels["trials"] = pa1.Xlabels["trials"].drop("tmp", axis=1)
        pa2.Xlabels["trials"] = pa2.Xlabels["trials"].drop("tmp", axis=1)

        return pa1, pa2

    def split_train_test_random(self, frac_test):
        """
        Quick, random split trials into train, test, taking random subset of trials.
        """
        import random
        ntrials = len(self.Trials)
        ntrials_sub = int(np.ceil(frac_test*ntrials))
        inds_all = list(range(len(self.Trials)))
        inds_test = sorted(random.sample(inds_all, ntrials_sub))
        inds_train = [i for i in inds_all if i not in inds_test]

        # print(len(inds_all), len(inds_train), len(inds_test))
        
        pa_train = self.slice_by_dim_indices_wrapper("trials", inds_train, reset_trial_indices=True)
        pa_test = self.slice_by_dim_indices_wrapper("trials", inds_test, reset_trial_indices=True)

        return pa_train, pa_test

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
        list_levels = sort_mixed_type(self.Xlabels[dim_str]["_dummy"].unique().tolist())

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
    def _dataextract_timewarp_piecewise_linear_trial(self, indtrial, anchor_times_this_trial, times_template, 
                                              anchor_times_template, smooth_boundaries_sigma=0.015, 
                                              PLOT=False, no_negative_fr_allowed=True):
        """
        For this trial, time-warp the neural data based on mathcing anchor pts to a template, and linearly warping between the
        anchor points.
        RETURNS:
        - a single PA holding this trial.
        """
        from neuralmonkey.utils.frmat import timewarp_piecewise_linear_interpolate

        X_trial = self.X[:, indtrial, :]
        times_trial = self.Times
        X_warped, fig1, fig2 = timewarp_piecewise_linear_interpolate(X_trial, times_trial, anchor_times_this_trial, times_template, 
                                          anchor_times_template, smooth_boundaries_sigma, PLOT)       

        if no_negative_fr_allowed:
            #  Make sure X is all positive. This is possible negative sometimes due to numerical imprecision and filtering?
            X_warped[X_warped<0] = 0.

        return X_warped, fig1, fig2

    def dataextract_timewarp_piecewise_linear(self, anchor_times_this_trial, times_template, 
                                              anchor_times_template, smooth_boundaries_sigma=0.015,
                                              PLOT=False):
        """
        Apply this timewarp to all trials, returning a copy of self which has all trials warped
        RETURNS:
        - PA, copy, shape (nchans, ntrials, len(times_template))
        """

        list_x = []
        fig1, fig2 = None, None
        for indtrial in range(self.X.shape[1]):
            if indtrial==0 and PLOT:
                X_warped, fig1, fig2 = self._dataextract_timewarp_piecewise_linear_trial(indtrial, anchor_times_this_trial, times_template, 
                                                                anchor_times_template, smooth_boundaries_sigma, PLOT=True)
            else:
                X_warped, _, _ = self._dataextract_timewarp_piecewise_linear_trial(indtrial, anchor_times_this_trial, times_template, 
                                                                anchor_times_template, smooth_boundaries_sigma, PLOT=False)
            list_x.append(X_warped)
        
        X = np.stack(list_x, axis=0)
        X = np.transpose(X, (1,0,2))
        

        PA = self.copy()
        PA.X = X
        PA.Times = times_template

        return PA, fig1, fig2

    def dataextract_reshape(self, reshape_method="chans_x_trialstimes"):
        """
        Holds methods for reshaping data.
        """
        if reshape_method=="chans_x_trialstimes":
            X = self.X.reshape(self.X.shape[0], -1)
        else:
            assert False

        return X

    def dataextract_as_distance_matrix_clusters_flex_reversed(self, var_group,
                                                     version_distance="euclidian",
                                                     accurately_estimate_diagonal=False):
        """
        A bit hacky variation of dataextract_as_distance_matrix_clusters_flex -- running that but on 
        call shuffles the times for one of the pairs of datapts being compoared -- this is null value if
        the timing of the neural trajectories does not matter.
        :return:
        - LIST_CLDIST, List of Cl, each holding distance of shape (ntrials, trials), if version_distance 
        is pairwise between pts, or shape (ngroups, ngroups), if version_distance is distribugtional (ie.
        is pairwise between levels of conjucntion of var_group)
        """
        from pythonlib.cluster.clustclass import Clusters
        
        assert version_distance == "euclidian", "only coded for this so far.."

        # Get labels info
        dflab = self.Xlabels["trials"]
        labels_rows = dflab.loc[:, var_group].values.tolist()
        labels_rows = [tuple(x) for x in labels_rows] # list of tuples

        # Get a shuffled version of self.X
        Xshuff = self.shuffle_by_time()
        
        # Option 1 - do independently for each time bin, and return list of all results.
        # - Collect Cldists, one for each time bin
        LIST_CLDIST = []
        LIST_TIME = []
        ntimes = self.X.shape[2]
        for i_time in range(ntimes):

            X1 = self.X[:, :, i_time].T # (trials, chans)
            X2 = Xshuff[:, :, i_time].T

            # Compute distance matrix directly
            # 0=idnetical, more positive more distance.
            from scipy.spatial import distance_matrix
            D = distance_matrix(X1, X2)

            params = {
                "version_distance":version_distance,
                "Clraw":None,
                "label_vars":var_group
            }
            Cldist = Clusters(D, labels_rows, labels_rows, ver="dist", params=params)

            if Cldist is None:
                print(D)
                assert False
            LIST_CLDIST.append(Cldist)
            LIST_TIME.append(self.Times[i_time])
        
        return LIST_CLDIST, LIST_TIME

    def dataextract_as_distance_index_between_two_base_classes(self, var_effect = "idx_morph_temp", effect_lev_base1=0, 
                                                               effect_lev_base2=99, list_grps_get=None, version="pts_time", 
                                                               PLOT=False, var_context_diff=None,
                                                               plot_conjunctions_savedir=None):
        """
        GOOD - to project all data onto 1d scalar mapping between two levels, <effect_lev_base1>, <effect_lev_base2>, of a variable <var_effect>.
        And many options to do so using singel trials (always against the groups of the base levels) or gorups, and time, vs. mean over time.

        NOTE: This uses raw euclidian distance to compute the metrics. 

        NOTE: Was previously _compute_df_using_dist_index_traj in psychometric...

        Map individual trials onto dist_index, which is based on eucl distance to base1 and base2.
        This does two special things:
        1. Works with trajectories - does each time bin one by one.
        2. Processes individual datapts (each vs. groups), instead of (groups vs. groups)

        NOTE:
        - DFPROJ_INDEX_AGG, with pts_or_groups=="pts" is (almoost) identical to DFPROJ_INDEX, with pts_or_groups=="grps", becuase the 
        former aggs over trials. Therefore, in general, using pts_or_groups=="pts" is better, but is slower.

        PARAMS:
        - var_context_diff, if not None, then only takes pairs of data that have different level for this var.
        
        EXAMPLE: Run this code to compute all the version, and to visualize that the following are equivalent:
        - pts --> grps, if you just agg over trials, using the outputed data
        - time --> scal, if you just agg over time, using the outputed data.

        var_effect = "idx_morph_temp"
        effect_lev_base1 = 0
        effect_lev_base2 = 99
        list_grps_get = [(0,), (99,)] # This is important, or else will fail if there are any (idx|assign) with only one datapt.


        version = "pts_time"
        DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                            effect_lev_base1, effect_lev_base2, 
                                                                                            version = version)
        DFPROJ_INDEX_SCAL = aggregGeneral(DFPROJ_INDEX, ["idx_row_datapt", "labels_1_datapt", "idx_morph_temp"], ["dist_index"])
        sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", kind="line", hue="idx_morph_temp")
        sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")

        version = "grps_time"
        DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                            effect_lev_base1, effect_lev_base2, 
                                                                                            version = version)
        DFPROJ_INDEX_SCAL = aggregGeneral(DFPROJ_INDEX, ["idx_morph_temp"], ["dist_index"])
        sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", kind="line", hue="idx_morph_temp")
        sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")


        version = "pts_scal"
        DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                            effect_lev_base1, effect_lev_base2, 
                                                                                            version = version)
        sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")

        version = "grps_scal"
        DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                            effect_lev_base1, effect_lev_base2, 
                                                                                            version = version)
        sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")

        """
        from neuralmonkey.scripts.analy_decode_moment_psychometric import dfdist_to_dfproj_index
        from neuralmonkey.scripts.analy_decode_moment_psychometric import dfdist_to_dfproj_index_datapts
        # At each time, score distance between pairs of groupigs 
        from pythonlib.tools.pandastools import aggregGeneral 
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good, grouping_plot_n_samples_conjunction_heatmap_helper

        if version=="pts_time":
            # Use indiv datapts (distnace between pts vs. groups), and separately for each time bin
            pts_or_groups="pts"
            return_as_single_mean_over_time=False
        elif version == "pts_scal":
            # Use indiv datapts, but use mean eucl distance across tiem bins. 
            # NOTE: this mean is taken of the eucl dist across time ibns, which themselves are computed independetmyl.
            # so this is nothing special. Is not that good - might as well run using pts_time, and then average the result over tmime.
            pts_or_groups="pts"
            return_as_single_mean_over_time=True
        elif version == "grps_time":
            # Distance bewteen (group vs. group), separately fro each time bin.
            pts_or_groups="grps"
            return_as_single_mean_over_time=False
        elif version == "grps_scal":
            # See above.
            pts_or_groups="grps"
            return_as_single_mean_over_time=True
        else:
            print(version)
            assert False, "typo for version?"

        if plot_conjunctions_savedir is not None:
            if var_context_diff is not None:
                fig = grouping_plot_n_samples_conjunction_heatmap_helper(self.Xlabels["trials"], [var_effect, var_context_diff])
            else:
                fig = grouping_plot_n_samples_conjunction_heatmap_helper(self.Xlabels["trials"], [var_effect])
            savefig(fig, f"{plot_conjunctions_savedir}/dataextract_as_distance_index_between_two_base_classes-final.pdf")
        plt.close("all")

        ### Get distance between all trials at each time bin
        version_distance = "euclidian"
        if return_as_single_mean_over_time:
            # Each trial pair --> scalar

            if var_context_diff is not None:
                assert False, "code it. see below, where return_as_single_mean_over_time=False"

            cldist = self.dataextract_as_distance_matrix_clusters_flex([var_effect], version_distance=version_distance,
                                                                    accurately_estimate_diagonal=False, 
                                                                    return_as_single_mean_over_time=return_as_single_mean_over_time)
            if pts_or_groups=="pts":
                # Score each datapt
                # For each datapt, get its distance to each of the groupings.
                # --> nrows = (ndatapts x n groups).
                # list_grps_get = [
                #     ("0|base1",),  
                #     ("99|base2",)
                #     ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.
                DFDIST = cldist.rsa_distmat_score_all_pairs_of_label_groups_datapts(list_grps_get=list_grps_get)
                DFPROJ_INDEX = dfdist_to_dfproj_index_datapts(DFDIST, var_effect=var_effect, 
                                                        effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
                # dfproj_index
                # order = sorted(dfproj_index["idxmorph_assigned_1"].unique())
                # sns.catplot(data=dfproj_index, x="idxmorph_assigned_1", y="dist_index", aspect=2, order=order)
            elif pts_or_groups=="grps":
                # Score pairs of (group, group)
                # Obsolete, because this is just above, followed by agging
                DFDIST = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

                # convert distnaces to distance index
                DFPROJ_INDEX = dfdist_to_dfproj_index(DFDIST, var_effect=var_effect, 
                                                        effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
            else:
                assert False

            # Take mean over trials
            if pts_or_groups=="pts":
                DFPROJ_INDEX_AGG = aggregGeneral(DFPROJ_INDEX, ["labels_1_datapt", var_effect], ["dist_index"])
                DFDIST_AGG = aggregGeneral(DFDIST, ["labels_1_datapt", "labels_2_grp", var_effect, f"{var_effect}_1", f"{var_effect}_2"], ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"])
            elif pts_or_groups == "grps":
                DFPROJ_INDEX_AGG = None
                DFDIST_AGG = None
            else:
                assert False

        else:
            # Each trial pair --> vector
    
            if var_context_diff is not None:
                _vars_grp = [var_effect, var_context_diff]
            else:
                _vars_grp = [var_effect]
            list_cldist, list_time = self.dataextract_as_distance_matrix_clusters_flex(_vars_grp, version_distance=version_distance,
                                                                                    accurately_estimate_diagonal=False, 
                                                                                    return_as_single_mean_over_time=return_as_single_mean_over_time)
            if var_context_diff is not None and list_grps_get is not None:
                # For each grp in list_grps_get, append each level of var_context_diff.
                # eg. if starting list_grps_get = [("0|base1",), ("99|base2",)], and var_context_diff="seqc_0_loc",
                # then modifies to list_grps_get = [('0|base1', (0, 0)), ('99|base2', (0, 0)), ('0|base1', (1, 1)), ('99|base2', (1, 1))]
                dflab = self.Xlabels["trials"]
                _levels = dflab[var_context_diff].unique().tolist()
                list_grps_get = [tuple(list(grp) + [_lev]) for _lev in _levels for grp in list_grps_get]

            ### For each time bin, for each trial, get its dist index relative to base1 and base2.
            list_dfproj_index = []
            list_dfdist = []
            for i, (cldist, time) in enumerate(zip(list_cldist, list_time)):
                if pts_or_groups=="pts":
                    # Score each datapt
                    # For each datapt, get its distance to each of the groupings.
                    # --> nrows = (ndatapts x n groups).
                    # list_grps_get = [
                    #     ("0|base1",),  
                    #     ("99|base2",)
                    #     ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.
                    if var_context_diff is not None:
                        # Then you only care about pairs that have different levels of var_context_diff...
                        ignore_self_distance = True
                    else:
                        ignore_self_distance = False
                    dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups_datapts(list_grps_get=list_grps_get,
                                                                                        ignore_self_distance=ignore_self_distance)

                    # display(dfdist)
                    # assert False, "confirm that levels are 2-tuples"

                    if var_context_diff is not None:
                        # Only keep cases where the datapt has different level of <var_context_diff> compared to the grp.
                        dfdist = dfdist[dfdist[f"{var_context_diff}_same"]==False].reset_index(drop=True)
                    dfproj_index = dfdist_to_dfproj_index_datapts(dfdist, var_effect=var_effect, 
                                                            effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
                    # dfproj_index
                    # order = sorted(dfproj_index["idxmorph_assigned_1"].unique())
                    # sns.catplot(data=dfproj_index, x="idxmorph_assigned_1", y="dist_index", aspect=2, order=order)
                elif pts_or_groups=="grps":
                    # Score pairs of (group, group)
                    # Obsolete, because this is just above, followed by agging

                    if var_context_diff is not None:
                        assert False, "code it. see above in pts_or_groups==pts. Beteter, just make this code"

                    dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

                    # convert distnaces to distance index
                    dfproj_index = dfdist_to_dfproj_index(dfdist, var_effect=var_effect, 
                                                            effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
                else:
                    assert False

                dfproj_index["time_bin"] = time
                dfdist["time_bin"] = time

                dfproj_index["time_bin_idx"] = i
                dfdist["time_bin_idx"] = i

                list_dfproj_index.append(dfproj_index)
                list_dfdist.append(dfdist)

            ### Clean up the results
            DFPROJ_INDEX = pd.concat(list_dfproj_index).reset_index(drop=True)
            DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
            # DFDIST[var_effect] = DFDIST[f"{var_effect}_1"]

            # display(DFPROJ_INDEX)
            # display(DFDIST)
            # assert False
            # Take mean over trials
            if pts_or_groups=="pts":
                DFPROJ_INDEX_AGG = aggregGeneral(DFPROJ_INDEX, ["labels_1_datapt", var_effect, "time_bin_idx"], ["dist_index"], nonnumercols=["time_bin"])
                if "dist_yue_diff" in DFDIST.columns:
                    values = ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff", "time_bin"]
                else:
                    values = ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "time_bin"]
                DFDIST_AGG = aggregGeneral(DFDIST, 
                                           ["labels_1_datapt", "labels_2_grp", var_effect, f"{var_effect}_1", 
                                                    f"{var_effect}_2", "time_bin_idx"], values)
            elif pts_or_groups == "grps":
                DFPROJ_INDEX_AGG = None
                DFDIST_AGG = None
            else:
                assert False

        # Also get dist index normalized to 0 and 1 (min and max)
        DFPROJ_INDEX["dist_index_norm"] = (DFPROJ_INDEX["dist_index"] - DFPROJ_INDEX["dist_index"].min())/(DFPROJ_INDEX["dist_index"].max() - DFPROJ_INDEX["dist_index"].min())
        if DFPROJ_INDEX_AGG is not None:
            DFPROJ_INDEX_AGG["dist_index_norm"] = (DFPROJ_INDEX_AGG["dist_index"] - DFPROJ_INDEX_AGG["dist_index"].min())/(DFPROJ_INDEX_AGG["dist_index"].max() - DFPROJ_INDEX_AGG["dist_index"].min())

        ######## GET DIFFERENCE ACROSS ADJACENT IDNICES
        if var_effect == "idx_morph_temp":
            from neuralmonkey.scripts.analy_decode_moment_psychometric import _rank_idxs_append, convert_dist_to_distdiff
            _rank_idxs_append(DFPROJ_INDEX)
            DFPROJ_INDEX_DIFFS = convert_dist_to_distdiff(DFPROJ_INDEX, "dist_index", "idx_morph_temp_rank")

            # Also get diff score normalized 
        else:
            DFPROJ_INDEX_DIFFS = None

        #### PLOT
        if PLOT:
            import seaborn as sns
            if version in ["pts_time", "grps_time"]:
                # DFPROJ_INDEX_SCAL = aggregGeneral(DFPROJ_INDEX, ["idx_row_datapt", "labels_1_datapt", "idx_morph_temp"], ["dist_index"])
                fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", kind="line", hue="idx_morph_temp")
                fig = sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")
                if DFPROJ_INDEX_DIFFS is not None:
                    sns.catplot(data=DFPROJ_INDEX_DIFFS, x="idx_along_morph", y="dist", kind="point")
            elif version in ["pts_scal", "grps_scal"]:
                fig = sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point")
                if DFPROJ_INDEX_DIFFS is not None:
                    sns.catplot(data=DFPROJ_INDEX_DIFFS, x="idx_along_morph", y="dist", kind="point")
            else:
                assert False

        trialcodes = self.Xlabels["trials"]["trialcode"].tolist()
        assert sorted(trialcodes)==sorted(set(DFPROJ_INDEX["trialcode"]))
        assert sorted(trialcodes)==sorted(set(DFDIST["trialcode"]))

        return DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS



    def dataextractwrap_distance_between_groups(self, vars_group, version):
        """
        Wrapper for all methods to get euclidian distnace between trial groups,
        defined by input variables.
        """

        if version=="traj_to_scal":
            # Compute for each time bin, then average over all time.
            Cldist = self.dataextract_as_distance_matrix_clusters_flex(vars_group, 
                                                                        return_as_single_mean_over_time=True)
            DFDIST = Cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

        elif version=="traj":
            # Compute for each time bin, keeping them seprate, thus returning time series.
            # Get pairwise dist between each shape at each timepoint
            list_cldist, list_time = self.dataextract_as_distance_matrix_clusters_flex(vars_group,
                                                                version_distance="euclidian",
                                                                agg_before_distance=False, 
                                                                return_as_single_mean_over_time=False)
            ### For each time bin, for each trial, get its dist index relative to base1 and base2.
            # list_dfproj_index = []
            list_dfdist = []
            for i, (cldist, time) in enumerate(zip(list_cldist, list_time)):
                print(time)
                
                # Score pairs of (group, group)
                dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=True)
                dfdist["time_bin"] = time
                dfdist["time_bin_idx"] = i

                list_dfdist.append(dfdist)

            ### Clean up the results
            DFDIST = pd.concat(list_dfdist).reset_index(drop=True)

        else:
            print(version)
            assert False

        return DFDIST

    def dataextract_as_distance_matrix_clusters_flex(self, var_group,
                                                     version_distance="euclidian",
                                                     accurately_estimate_diagonal=False,
                                                     agg_before_distance=False,
                                                     return_as_single_mean_over_time=False):
        """
        GOOD - Extract distance matrix between trials, with flexible ways of c,m,puting and agging over variables.
        :params: var_group,list, determines the trial labels kept in Cl, and, if vd is distributional, then
        how to group trials into distributiosn
        :params: accurately_estimate_diagonal, bool, True means faster compute. In general I don't use
        the diagonal, so leave false.
        :agg_before_distance: bool, only applies for version_distance that is not distributional. By defautl (FAlse),
        returns distance between trials. If True, then first agg so returns shape (n levels of var_group, n levels of var_group)
        :return:
        - LIST_CLDIST, List of Cl, each holding distance of shape (ntrials, trials), if version_distance 
        is pairwise between pts, or shape (ngroups, ngroups), if version_distance is distribugtional (ie.
        is pairwise between levels of conjucntion of var_group)
        """
        from pythonlib.cluster.clustclass import Clusters

        if version_distance == "euclidian":
            version_distance_is_not_distributional = True
        elif version_distance == "euclidian_unbiased":
            version_distance_is_not_distributional = False
        else:
            print(version_distance)
            assert False, "fill this in"
        
        # version_distance_is_not_distributional = version_distance=="euclidian"

        if agg_before_distance and version_distance_is_not_distributional:
            PA = self.slice_and_agg_wrapper("trials", var_group)
        else:
            PA = self.copy()

        # Option 1 - do independently for each time bin, and return list of all results.
        # - Collect Cldists, one for each time bin
        LIST_CLDIST = []
        LIST_TIME = []
        ntimes = PA.X.shape[2]
        print("... computing distance matrices, using distnace:", version_distance)
        for i_time in range(ntimes):
            
            # make a pa that just has this one time bin
            pa_single_time = PA.slice_by_dim_indices_wrapper("times", [i_time, i_time])
            assert pa_single_time.X.shape[2]==1
            
            # Create clusters
            dflab = pa_single_time.Xlabels["trials"]
            Xthis = pa_single_time.X.squeeze(axis=2).T # (ntrials, ndims)
            
            # print("  Final Scalar data (trial, dims):", Xthis.shape)
            label_vars = var_group
            labels_rows = dflab.loc[:, label_vars].values.tolist()
            labels_rows = [tuple(x) for x in labels_rows] # list of tuples
            params = {"label_vars":label_vars}

            trialcodes = dflab["trialcode"].tolist()
            if False:
                # Compute distance matrix directly
                # 0=idnetical, more positive more distance.
                from scipy.spatial import distance_matrix
                D = distance_matrix(Xthis, Xthis)

                params = {
                    "version_distance":version_distance,
                    "Clraw":None,
                }
                Cldist = Clusters(D, labels_rows, labels_rows, ver="dist", params=params)

            else:
                # Compute using Cl intermediate
                Cl = Clusters(Xthis, labels_rows, ver="rsa", params=params, trialcodes=trialcodes)
                            
                # convert to distance matrix
                if version_distance == "euclidian":
                    Cldist = Cl.distsimmat_convert(version_distance)
                else:
                    Cldist = Cl.distsimmat_convert_distr(label_vars, version_distance, accurately_estimate_diagonal=accurately_estimate_diagonal)
            
            LIST_CLDIST.append(Cldist)
            LIST_TIME.append(PA.Times[i_time])

        # Sanity check that all Cl match
        labels = None
        labels_cols = None
        for t, Cldist in zip(LIST_TIME, LIST_CLDIST):
            # check labels match
            if labels is not None:
                assert labels == Cldist.Labels
                labels = Cldist.Labels
            if labels_cols is not None:
                assert labels_cols == Cldist.LabelsCols
                labels_cols = Cldist.LabelsCols

        if return_as_single_mean_over_time:
            ### Take mean distance over time, and construct a single Clusters
            Xinpput_mean = np.mean(np.stack([Cldist.Xinput for Cldist in LIST_CLDIST], axis=0), axis=0)
            params = {
                "label_vars":LIST_CLDIST[0].Params["label_vars"],
                "version_distance":LIST_CLDIST[0].Params["version_distance"],
                "Clraw":None,
            }
            list_lab = LIST_CLDIST[0].Labels
            Cldist = Clusters(Xinpput_mean, list_lab, list_lab, ver="dist", params=params, trialcodes = LIST_CLDIST[0].Trialcodes)
            return Cldist
        else:
            return LIST_CLDIST, LIST_TIME

    def dataextract_pca_demixed_subspace(self, var_pca, vars_grouping,
                                                pca_twind, pca_tbindur, # -- PCA params start
                                                pca_filtdict=None, savedir_plots=None,
                                                raw_subtract_mean_each_timepoint=True,
                                                pca_subtract_mean_each_level_grouping=True,
                                                n_min_per_lev_lev_others=4, prune_min_n_levs = 2,
                                                n_pcs_subspace_max = None, 
                                                do_pca_after_project_on_subspace=False,
                                                PLOT_STEPS=False, SANITY=False,
                                                reshape_method="trials_x_chanstimes",
                                                pca_tbin_slice=None, return_raw_data=False,
                                                proj_twind=None, proj_tbindur=None, proj_tbin_slice=None, # --- Extra params for projecting data to PC space
                                                inds_pa_fit=None, inds_pa_final=None
                                                ):
        """
        Helper to construct pca space (deminxed pca) in flexible ways, and then project raw data onto this space.
        Uses representations that are sliced data (n sub slices, concate into vector).

        Does work of projecting raw data onto this space.

        Example:
            # This does pca on means for stroke_index, controlling for context defined by each level of vars_grouping
            var_pca = "stroke_index"
            vars_grouping = ["task_kind", "gridloc", "shape"]
            filtdict= None

        Example:
            # This does pca on means for each level of conjunctive var_pca.
            var_pca = ("gridloc", "stroke_index")
            vars_grouping = None
            filtdict= None

        Can do this for both scalar (mean over time, returns (ntrials, npcs) and trajectories (retains time, returns (npcs, ntrials, ntimes)),
        --- scalar, make reshape_method "trials_x_chanstimes"
        --- trajs, make reshape_method "chans_x_trials_x_times"

        :param var_pca: str, the variable to find subspaces for, takes mean over these after subtracting out and conditioning
        on each levle of vars_goruping, then does PCA.
        :param vars_grouping: list of str, see var_pca.
        :param pca_twind: e.g, (-0.1, 0.1), entire window of data to slice out
        :param pca_tbindur: time sec, width of sliding window wiwthin pca_twind. e.g, if pca_twind==(-0.1, 0.1) and pca_tbindur
        is 0.1, then dimension of raw data is nchans*2.
        :param filtdict:
        :param raw_subtract_mean_each_timepoint:
        :param pca_subtract_mean_each_level_grouping:
        :param n_pcs_subspace_max, int, keeps maximum this many of the top dimensions after projecting into subspace.
        :param do_pca_after_project_on_subspace, bool, if True, does on final after proj. Useful if want to rotate
        final data but this usually shouldnt be done.
        :param PLOT_STEPS:
        :param return_raw_data: bool, return data that has not yet been sliced for pca.
        :param inds_pa_fit, inds_pa_final, either None (Ignore) or list of ints, which are the rows of self, for splitting data into train (for
        fitting PCA) and test (for projecting i))
        :return:
        - Xredu, (ntrials, npc dims)
        - Xfinal, (ntrials, nchan * timesteps), data immed before pCA
        - PAfinal, holds Xfinal
        - PAraw, holds data in raw dimensions, but with all prepprocess done -- ie can project this into pc space
        - pca, dict, holds PC features.

        RETURNS None if no data found for this var_pca
        """
        from neuralmonkey.analyses.state_space_good import dimredgood_pca_project

        # if savedir_plots is None:
        #     savedir_plots = "/tmp"

        # If the inputed variables are not exist, then return None
        if var_pca not in self.Xlabels["trials"].columns:
            return  (None for _ in range(5))
        if vars_grouping is not None:
            for var in vars_grouping:
                if var not in self.Xlabels["trials"].columns:
                    return (None for _ in range(5))

        ###### Prep variables
        if n_pcs_subspace_max is None:
            n_pcs_subspace_max = 10

        # Params for windowing data before projecting to pc space. By default use the same params as 
        # for fitting PCA.
        if proj_twind is None:
            proj_twind = pca_twind
        if proj_tbindur is None:
            proj_tbindur = pca_tbindur
        if proj_tbin_slice is None:
            proj_tbin_slice = pca_tbin_slice

        if isinstance(var_pca, (list, tuple)):
            # conjunctive variable...
            from pythonlib.tools.pandastools import append_col_with_grp_index
            self.Xlabels["trials"] = append_col_with_grp_index(self.Xlabels["trials"], var_pca, "|".join(var_pca))
            var_pca = "|".join(var_pca)

        if reshape_method=="chans_x_trials_x_times":
            dimredgood_pca_project_do_reshape = True
        else:
            dimredgood_pca_project_do_reshape = False
            
        if pca_tbin_slice is None:
            pca_tbin_slice = pca_tbindur

        # For plotting exmaples...
        chan = self.Chans[0]

        if PLOT_STEPS:
            self.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        ############### ALL NORMALIZING ON RAW DATA, BEFORE take avg and do pca, so that can pass in raw and project to pc space.
        PAraw = self.copy()
        assert PAraw.X.shape[1]>0, "intpou has no trials"

        if raw_subtract_mean_each_timepoint:
            PAraw = PAraw.norm_subtract_trial_mean_each_timepoint()
            if PLOT_STEPS:
                PAraw.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        ############### Perform PCA -- (1) Extract data for computing projection
        # Split into train/test?
        if inds_pa_fit is not None:
            assert inds_pa_final is not None
            pa = PAraw.slice_by_dim_indices_wrapper("trials", inds_pa_fit) # fewer trials usually
            PAraw = PAraw.slice_by_dim_indices_wrapper("trials", inds_pa_final)

            # Devo -- doing train/test split here.
            # # Split PA into fit and held-out
            # test_size = 0.1
            # if vars_grouping is not None:
            #     _vars_grp = [var_pca] + vars_grouping
            # else:
            #     _vars_grp = [var_pca]
            # PAraw, pa = PAraw.split_sample_stratified_by_label(_vars_grp, test_size=test_size, PRINT=False)
            # print(PAraw.X.shape)
            # print(pa.X.shape)
            # print(n_min_per_lev_lev_others)
            # n_min_per_lev_lev_others = 1

        else:
            # Fit using same data.
            pa = PAraw.copy()

        # Apply filtdict
        pa = pa.slice_by_labels_filtdict(pca_filtdict)

        # Keep only othervar that has multiple cases of var
        if savedir_plots is not None:
            plot_counts_heatmap_savepath = f"{savedir_plots}/counts_conj.pdf"
        else:
            plot_counts_heatmap_savepath = None

        pa, dfout, dict_dfthis = pa.slice_extract_with_levels_of_conjunction_vars(var_pca, vars_grouping, n_min_per_lev_lev_others,
                                                              prune_min_n_levs, plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
        if pa is None:
            print("No variation found for this var_pca (reutrning None): ", var_pca)
            return (None for _ in range(5))
        if PLOT_STEPS:
            pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        # count, what is the max n lev across all otherlevs
        if False:
            # get the max within lev (across levs)
            print([len(df[var_pca].unique()) for df in dict_dfthis.values()])
        else:
            # get n classes after lumping all olev
            nclasses_of_var_pca = len(dfout[var_pca].unique())

        # Subtract mean for the variables you want to "condition on"
        if pca_subtract_mean_each_level_grouping:
            pa = pa.norm_by_label_subtract_mean("trials", vars_grouping)
            if PLOT_STEPS:
                pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        # Get new PA that averages to get one value for each state (var x vars_groupig)
        if vars_grouping is None:
            grouping_variables = [var_pca]
        else:
            grouping_variables = [var_pca]+vars_grouping
        pa = pa.slice_and_agg_wrapper("trials", grouping_variables)
        if PLOT_STEPS:
            pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        # Finally, demean again
        if False: # dont do this, since this leads to separation of each context again...
            pa = pa.norm_subtract_trial_mean_each_timepoint()
            # print(pa.X.shape)
            # print(np.mean(pa.X[:, :, 10], axis=1))
            # assert False
            if PLOT_STEPS:
                pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var_pca, vars_grouping)

        ############### Perform PCA -- (2) Fit PCA
        # Do PCA for each time window (normalize within that window)
        if savedir_plots is not None:
            plot_pca_explained_var_path = f"{savedir_plots}/expvar.pdf"
            plot_loadings_path = f"{savedir_plots}/loadings.pdf"
        else:
            plot_pca_explained_var_path = None
            plot_loadings_path = None

        assert pa.X.shape[0] == PAraw.X.shape[0]
        assert pa.X.shape[2] == PAraw.X.shape[2], "or else projection will fail later"

        _, PApca, _, pca, X_before_dimred = pa.dataextract_state_space_decode_flex(pca_twind, pca_tbindur, pca_tbin_slice,
                                                          reshape_method=reshape_method, pca_reduce=True,
                                                          plot_pca_explained_var_path=plot_pca_explained_var_path,
                                                          plot_loadings_path=plot_loadings_path, how_decide_npcs_keep="cumvar",
                                                          pca_frac_var_keep=0.95, 
                                                        norm_subtract_single_mean_each_chan=False)
        pca["explained_variance_ratio_initial_construct_space"] = pca["explained_variance_ratio_"]
        del pca["explained_variance_ratio_"]
        pca["X_before_dimred"] = X_before_dimred
        pca["nclasses_of_var_pca"] = nclasses_of_var_pca
        # print("----")
        # print(pa.X.shape)
        # print(PApca.X.shape)
        # print(pca["X_before_dimred"].shape)
        # print(pca["components"].shape)
        # assert False
        # assert False

        if SANITY:
            from neuralmonkey.analyses.state_space_good import dimredgood_pca_project
            X = pca["X_before_dimred"]
            dimredgood_pca_project(pca["components"], X, "/tmp/test2", 
                                   do_additional_reshape_from_ChTrTi=dimredgood_pca_project_do_reshape)
            print("HCECK tmp/test2 -- This should match the exp var above...")
            assert False

        # Plot results in state space
        if False: # Need to use trajectories...
            if savedir_plots is not None:
                from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER
                ndims = PApca.X.shape[0]
                if ndims<4:
                    list_dims = [(0,1)]
                else:
                    list_dims = [(0,1), (1,2)]

                dflab = PApca.Xlabels["trials"]
                xthis = PApca.X.squeeze(axis=2).T # (n4trials, ndims)
                save_suffix = "DAT_GO_INTO_PCA"
                trajgood_plot_colorby_splotby_scalar_WRAPPER(xthis, dflab, var_pca, savedir_plots,
                                                            vars_subplot=None, list_dims=list_dims,
                                                            skip_subplots_lack_mult_colors=False,
                                                            save_suffix=save_suffix)
                trajgood_plot_colorby_splotby_scalar_WRAPPER(xthis, dflab, var_pca, savedir_plots,
                                                            vars_subplot=vars_grouping, list_dims=list_dims,
                                                            skip_subplots_lack_mult_colors=False,
                                                            save_suffix=save_suffix)

        ########### (3) Project RAW data back into this space
        # Figure out how many dimensions to keep (for euclidian).
        n1 = pca["nclasses_of_var_pca"] # num classes of superv_dpca_var that exist. this is upper bound on dims.
        if n1<4:
            n1 = 4
        if False:
            n2 = PApca.X.shape[1] # num classes to reach criterion for cumvar for pca.
            if n2<4:
                n2 = 4 # Then ignore it. Someitnes is very low D, but still want to keep dims >0
        else:
            n2 = 10000
        n3 = PApca.X.shape[0] # num dimensions.
        if n3<4:
            n3 = 4
        n_pcs_subspace_max = min([n1, n2, n3, n_pcs_subspace_max])

        # PApca = PApca.slice_by_dim_indices_wrapper("chans", list(range(n_pcs_keep_euclidian)))
        
        if True:
            Xredu, PAredu, stats_redu, Xfinal_before_redu, pca = PAraw.dataextract_pca_demixed_subspace_project(pca, proj_twind, 
                                                proj_tbindur, proj_tbin_slice, reshape_method,
                                                dimredgood_pca_project_do_reshape, n_pcs_subspace_max, do_pca_after_project_on_subspace,
                                                savedir_plots)
        else:
            # - preprocess data
            Xfinal_before_redu, PAfinal_before_redu, _, _, _ = PAraw.dataextract_state_space_decode_flex(pca_twind, pca_tbindur, pca_tbin_slice,
                                                                                    reshape_method=reshape_method, pca_reduce=False,
                                                                                    norm_subtract_single_mean_each_chan=False)
            
            ### Project all raw data
            plot_pca_explained_var_path = f"{savedir_plots}/expvar_reproj_raw.pdf"
            Xredu, stats_redu, Xredu_in_orig_shape = dimredgood_pca_project(pca["components"], 
                                                                            Xfinal_before_redu, 
                                                                            plot_pca_explained_var_path=plot_pca_explained_var_path,
                                                                            do_additional_reshape_from_ChTrTi=dimredgood_pca_project_do_reshape)

            if dimredgood_pca_project_do_reshape:
                # TRAJ, returns (ndims, ntrials,  ntimes)
                del Xredu

                # Keep only n final dimensions in subspace
                if n_pcs_subspace_max is not None and n_pcs_subspace_max <= Xredu_in_orig_shape.shape[0]:
                    Xredu_in_orig_shape = Xredu_in_orig_shape[:n_pcs_subspace_max, :, :]

                if do_pca_after_project_on_subspace:
                    assert False, "not yet coded for trajecroreis"
                    # Optionally, do PCA again on raw data that was projected into this subspace (useful for visualization).
                    # [IGNORE -- this is counterproductive]. This is just for rotation...
                    from neuralmonkey.analyses.state_space_good import dimredgood_pca
                    plot_pca_explained_var_path_this = f"{savedir_plots}/expvar_pca_after_subspace_projection.pdf"
                    plot_loadings_path_this = f"{savedir_plots}/loadings_pca_after_subspace_projection.pdf"
                    Xredu, _, _ = dimredgood_pca(Xredu, how_decide_npcs_keep = "keep_all",
                                    plot_pca_explained_var_path=plot_pca_explained_var_path_this,
                                plot_loadings_path=plot_loadings_path_this)

                # Get a PA holding final projected data
                PAredu = PAfinal_before_redu.copy_replacing_X(Xredu_in_orig_shape)

                # print(Xredu_in_orig_shape.shape) # (npcs, ntrials, ntimes)
                # print(PAredu.X.shape) # (npcs, ntrials, ntimes)
                # print(Xfinal_before_redu.shape) # (nchans, ntrials, ntimes)
                # print(pca["X_before_dimred"].shape) # (nchans, ngroups, ntimes)
                # assert False
                return Xredu_in_orig_shape, PAredu, stats_redu, Xfinal_before_redu, pca

            else:
                # SCALAR, returns (ntrials, ndims)

                # Keep only n final dimensions in subspace
                if n_pcs_subspace_max is not None and n_pcs_subspace_max <= Xredu.shape[1]:
                    Xredu = Xredu[:, :n_pcs_subspace_max]

                if do_pca_after_project_on_subspace:
                    # Optionally, do PCA again on raw data that was projected into this subspace (useful for visualization).
                    # [IGNORE -- this is counterproductive]. This is just for rotation...
                    from neuralmonkey.analyses.state_space_good import dimredgood_pca
                    plot_pca_explained_var_path_this = f"{savedir_plots}/expvar_pca_after_subspace_projection.pdf"
                    plot_loadings_path_this = f"{savedir_plots}/loadings_pca_after_subspace_projection.pdf"
                    Xredu, _, _ = dimredgood_pca(Xredu, how_decide_npcs_keep = "keep_all",
                                    plot_pca_explained_var_path=plot_pca_explained_var_path_this,
                                plot_loadings_path=plot_loadings_path_this)

                # Get a PA holding final projected data
                assert len(PAfinal_before_redu.Xlabels["trials"]) == Xredu.shape[0]
                PAredu = PopAnal(Xredu.T[:, :, None].copy(), [0])  # (ndimskeep, ntrials, 1)
                PAredu.Xlabels = {dim:df.copy() for dim, df in PAfinal_before_redu.Xlabels.items()}

                # print(Xredu.shape)
                # print(PAredu.X.shape)
                # print(Xfinal_before_redu.shape)
                # print(pca["X_before_dimred"].shape)
                # assert False
                return Xredu, PAredu, stats_redu, Xfinal_before_redu, pca
        
        if return_raw_data:
            return Xredu, PAredu, stats_redu, Xfinal_before_redu, pca, PAraw
        else:
            return Xredu, PAredu, stats_redu, Xfinal_before_redu, pca

    def dataextract_smooth_gaussian(self, std_sec, plot=False):
        """
        Smooth FR over time, returning a copy.
        Prunes the onset and offset, so the times will be subset of input times.
        PARAMS:
        - std_sec, std of gaussian kernel, in sec.
        """

        # Convolve spikes to smooth them
        from scipy.signal.windows import gaussian
        import matplotlib.pyplot as plt
        from scipy.signal import lfilter, filtfilt

        # Scale the width of the Gaussian by our bin width
        bin_width_sec = np.mean(np.diff(self.Times))

        std = int(std_sec / bin_width_sec)
        if std==0:
            std = 1

        # We need a window length of 3 standard deviations on each side (x2)
        M = std * 3 * 2
        window = gaussian(M, std)
        # Normalize so the window sums to 1
        window = window / window.sum()

        if plot:
            _ = plt.stem(window)
            
        # Remove convolution artifacts
        invalid_len = len(window) // 2
        # Xfilt = lfilter(window, 1, self.X, axis=2)[:, :, invalid_len:]
        # times_filt = self.Times[invalid_len:]

        pad = int(invalid_len/2)
        Xfilt = filtfilt(window, 1, self.X, axis=2)[:, :, pad:-pad]
        times_filt = self.Times[pad:-pad]
                
        assert Xfilt.shape[2] == len(times_filt), f"Xfilt shape: {Xfilt.shape}, times shape: {len(times_filt)}"

        # Visualize raw spikes, smoothed spikes, and PSTHs
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharey=True)
            _ = ax1.imshow(self.X[:, 0, :])
            _ = ax2.imshow(Xfilt[:, 0, :])
            ax2.set_title("One example trial")
            ax2.set_xlabel("time")
            ax2.set_ylabel("chan")
            plt.tight_layout()

            fig, ax = plt.subplots()
            ax.plot(self.Times, self.X[0, 0, :])
            ax.plot(times_filt, Xfilt[0, 0, :])

            fig, ax = plt.subplots()
            ax.plot(self.Times, self.X[10, 2, :])
            ax.plot(times_filt, Xfilt[10, 2, :])

        # create new PA
        return self.copy_replacing_X(Xfilt, times_filt)
        
    def dataextract_dimred_wrapper(self, scalar_or_traj, dim_red_method, savedir, 
                                   twind_pca, tbin_dur=None, tbin_slide=None, 
                                   NPCS_KEEP = 10,
                                   dpca_var = None, dpca_vars_group = None, dpca_filtdict=None, dpca_proj_twind = None, 
                                   raw_subtract_mean_each_timepoint=False,
                                   umap_n_components=2, umap_n_neighbors=40,
                                   inds_pa_fit=None, inds_pa_final=None,
                                   n_min_per_lev_lev_others = 2,
                                   return_pca_components=False):
        """
        THE Wrapper for all often-used methods for dim reduction

        PARAMS:
        - scalar_or_traj, str, either "scal" [returns (dims, trials, 1)], or "traj" [returns (dims, trials, timebins)]
        - version, str, either "pca" or "dpca"
        - twind_pca, (t1, t2), window to use for pca (for fitting)
        - tbin_dur, in sec, for smoothing
        - tbin_slide, in sec, smoothign
        - NPCS_KEEP, int
        - dpca_var, str
        - dpca_vars_group, list of str
        - dpca_proj_twind = None, window for the final data.
        - raw_subtract_mean_each_timepoint=False
        RETURNS:
        - Xredu, 
        --- if scalar_or_traj==scal --> (ntrials, ndims)
        --- if scalar_or_traj==traj --> (ndims, ntrials, ntimes)
        - PAredu, holding reduced data.
        """

        PA = self

        if umap_n_components is None:
            umap_n_components=2
        if umap_n_neighbors is None:
            umap_n_neighbors=40

        # Extra dim reductions?
        METHOD = "basic"
        if dim_red_method is None:
            # Then use raw data
            pca_reduce = False
            extra_dimred_method = None
            extra_dimred_method_n_components = None
        elif dim_red_method=="pca":
            pca_reduce = True
            extra_dimred_method = None
            extra_dimred_method_n_components = None
        elif dim_red_method=="pca_proj":
            # Use one window to fit PC space, then project larger window onto that space.
            pca_reduce = True
            extra_dimred_method = None
            extra_dimred_method_n_components = None
            assert twind_pca is not None, "this is the data for fitting PC space"
            assert dpca_proj_twind is not None, "This is final window size"
            # assert not twind_pca == dpca_proj_twind, "yo should just do dim_red_method=pca"
        elif dim_red_method=="pca_umap":
            # PCA --> UMAP
            pca_reduce = True
            extra_dimred_method = "umap"
            extra_dimred_method_n_components = umap_n_components
        elif dim_red_method=="umap":
            # UMAP
            pca_reduce = False
            extra_dimred_method = "umap"
            extra_dimred_method_n_components = umap_n_components
        elif dim_red_method=="mds":
            # MDS
            pca_reduce = False
            extra_dimred_method = "mds"
            extra_dimred_method_n_components = umap_n_components
        elif dim_red_method in ["dpca", "superv_dpca"]:
            # Supervised, based on DPCA, find subspace for a given variable by doing PCA on the mean values.
            assert dpca_var is not None
            if dpca_vars_group is not None:
                assert isinstance(dpca_vars_group, (list, tuple))
            # superv_dpca_var = superv_dpca_params["superv_dpca_var"]
            # superv_dpca_vars_group = superv_dpca_params["superv_dpca_vars_group"]
            # superv_dpca_filtdict = superv_dpca_params["superv_dpca_filtdict"]
            METHOD = "dpca"
        else:
            print(dim_red_method)
            assert False
        
        if tbin_dur is None:
            # Assume you want to take all time bins... None takes time average..
            tbin_dur = "ignore"
        if tbin_slide is None:
            # If conitnue with None, then code  makes it equal to tbin_dur
            tbin_slide = 0.01

        if scalar_or_traj in ["traj", "trajectory"]:
            reshape_method = "chans_x_trials_x_times"
            if tbin_dur == "default":
                tbin_dur = 0.15
                tbin_slide = 0.02
        elif scalar_or_traj in ["scal", "scalar"]:
            reshape_method = "trials_x_chanstimes"
            if dpca_proj_twind is not None:
                # Force it to be the same
                dpca_proj_twind = None
                # assert twind_pca == dpca_proj_twind, "scalar assumes a fixed time window, or else n features in pc will not match when try to project new data."
            if tbin_dur == "default":
                tbin_dur = 0.2
                tbin_slide = 0.1
        else:
            print(scalar_or_traj)
            assert False

        if METHOD=="basic":
            # Then just PCA on all data.

            # - normalize - remove time-varying component
            if raw_subtract_mean_each_timepoint:
                PA = PA.norm_subtract_trial_mean_each_timepoint()
            
            # - PCA
            if savedir is not None:
                plot_pca_explained_var_path=f"{savedir}/pcaexp.pdf"
                plot_loadings_path = f"{savedir}/pcaload.pdf"
            else:
                plot_pca_explained_var_path = None
                plot_loadings_path = None

            if dim_red_method=="pca_proj":
                # Save a copy of PA
                PAraw = PA.copy()
            else:
                # Sanity checks about times
                if dpca_proj_twind is not None:
                    if not dpca_proj_twind==twind_pca:
                        print(twind_pca, dpca_proj_twind)
                        assert False, "you entered params as if you wanted to use diff twind to fit pca (twind_pca) vs the final data size (dpca_proj_twind), but using diff windows only works for dataextract_pca_demixed_subspace. You should avoid entering dpca_proj_twind if doing pca"

            Xredu, PAredu, _, pca, X_before_dimred = PA.dataextract_state_space_decode_flex(twind_pca, tbin_dur, tbin_slide, reshape_method=reshape_method,
                                                        pca_reduce=pca_reduce, plot_pca_explained_var_path=plot_pca_explained_var_path, 
                                                        plot_loadings_path=plot_loadings_path, npcs_keep_force=NPCS_KEEP,
                                                        extra_dimred_method_n_components = extra_dimred_method_n_components,
                                                        extra_dimred_method=extra_dimred_method, umap_n_neighbors = umap_n_neighbors)    

            if dim_red_method=="pca_proj":
                pca["explained_variance_ratio_initial_construct_space"] = pca["explained_variance_ratio_"]
                del pca["explained_variance_ratio_"]
                pca["X_before_dimred"] = X_before_dimred
                # pca["nclasses_of_var_pca"] = nclasses_of_var_pca

                if reshape_method=="chans_x_trials_x_times":
                    dimredgood_pca_project_do_reshape = True
                else:
                    dimredgood_pca_project_do_reshape = False


                if False: # SANITY
                    from neuralmonkey.analyses.state_space_good import dimredgood_pca_project
                    X = pca["X_before_dimred"]
                    print(X.shape)
                    print(PAraw.X.shape)
                    dimredgood_pca_project(pca["components"], X, "/tmp/test2", 
                                        do_additional_reshape_from_ChTrTi=dimredgood_pca_project_do_reshape)
                    print("HCECK tmp/test2 -- This should match the exp var above...")
                    assert False


                Xredu, PAredu, _, _, _ = PAraw.dataextract_pca_demixed_subspace_project(pca, dpca_proj_twind, 
                                                    tbin_dur, tbin_slide, reshape_method,
                                                    dimredgood_pca_project_do_reshape, NPCS_KEEP, False, savedir)

            # n_pcs_keep_euclidian = PAredu.X.shape[1]

        elif METHOD=="dpca":
            # Then does targeted dim reduction, first averaging over trials to get means for variable.

            assert dpca_var is not None
            # n_min_per_lev_lev_others = 4
            Xredu, PAredu, _, _, pca = PA.dataextract_pca_demixed_subspace(dpca_var, dpca_vars_group,
                                                            twind_pca, tbin_dur, # -- PCA params start
                                                            dpca_filtdict,
                                                            pca_tbin_slice = tbin_slide,
                                                            savedir_plots=savedir,
                                                            raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                            pca_subtract_mean_each_level_grouping=True,
                                                            n_min_per_lev_lev_others=n_min_per_lev_lev_others, prune_min_n_levs = 2,
                                                            n_pcs_subspace_max = NPCS_KEEP, 
                                                            reshape_method=reshape_method,
                                                            proj_twind=dpca_proj_twind,
                                                            inds_pa_fit=inds_pa_fit, inds_pa_final=inds_pa_final)
        else:
            print(dim_red_method)
            assert False

        if return_pca_components:
            return Xredu, PAredu, pca
        else:
            return Xredu, PAredu

    def dataextract_subspace_targeted_pca_project(self, dfcoeff, list_subspace_tuples, normalization=None, 
                                                  plot_orthonormalization=False):
        """
        Project self data onto subspace defined by axes which are columns in dfcoeff.
        PARAMS:
        - dfcoeff, columns are names of axes.
        - list_subspace_tuples, list of tuples, each tuple is a subspace. Each subspace is a list of variables, which are the 
        columns of dfcoeff, to pick out the axes to project onto.
        """

        # Get basis vectors
        dict_subspace_pa = {}
        dict_subspace_axes_orig = {}
        dict_subspace_axes_normed = {}
        for subspace_tuple in list_subspace_tuples:

            basis_vectors_orig = dfcoeff.loc[:, subspace_tuple].values # (nchans, nrank)
            PAredu, basis_vectors_normed = self.dataextract_project_data_denoise(basis_vectors_orig, normalization=normalization, 
                                                         plot_orthonormalization=plot_orthonormalization)
            dict_subspace_pa[tuple(subspace_tuple)] = PAredu
            dict_subspace_axes_orig[tuple(subspace_tuple)] = basis_vectors_orig # in original space.
            dict_subspace_axes_normed[tuple(subspace_tuple)] = basis_vectors_normed # in original space.

            # For example
            # print(PAredu.X.shape) # (2, 356, 1)
            # print(PA.X.shape) (297, 356, 1)
            # print(basis_vectors.shape) (297, 2)

        if False:        
            for k, v in dict_subspace_pa:
                print(k, " -- ", type(k[0]), " -- ", v)

        return dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed

    def dataextract_subspace_targeted_pca_one_axis_per_var(self, variables, variables_is_cat, list_subspaces, demean=True, 
                                          normalization=None, plot_orthonormalization=False, 
                                          PLOT_COEFF_HEATMAP=False, savedir_coeff_heatmap=None, PRINT=False,
                                          get_axis_for_categorical_vars=True):
        """
        [GOOD] Get subspace for a set of variables, and then project data into that subspace.
        one_axis_per_var --> returns projected so that each var gets one dimension. e.g., project onto a 3d space
        defined by (shape, location, epoch)

        PARAMS:
        - variables, list of str, the variables that will be fit using regression, and from which the regression coefficients
        will be used to get the subspace. REgression performed independently for each neuron.
        PARAMS:
        - variables, list of str, the variables that will be fit using regression, and from which the regression coefficients
        - variables_is_cat, list of bool, whether each variable is categorical or not.
        - list_subspaces, list of tuples, each tuple is a subspace. Each subspace is a list of variables.
        - normalization, str, either None, "norm", or "orthonormalize".
        """ 

        for subspace_tuple in list_subspaces:
            assert isinstance(subspace_tuple, (tuple, list))
            assert isinstance(subspace_tuple[0], str)

        # Input must be scalarized
        assert self.X.shape[2]==1

        # Demean 
        if demean:
            PA = self.norm_subtract_mean_each_chan()
        else:
            PA = self

        ### Collect coefficients across all neurons
        dfcoeff, dfbases, res_all, original_feature_mapping = PA.regress_neuron_task_variables_all_chans(variables, variables_is_cat, PLOT_COEFF_HEATMAP, 
                                                                      PRINT=PRINT, savedir_coeff_heatmap=savedir_coeff_heatmap,
                                                                      get_axis_for_categorical_vars=get_axis_for_categorical_vars)
        

        # Get basis vectors
        dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed = PA.dataextract_subspace_targeted_pca_project(
            dfcoeff, list_subspaces, normalization, plot_orthonormalization)

        return dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed, dfcoeff, PA

    def dataextract_subspace_targeted_pca_one_var_mult_axes(self, variables, variables_is_cat, var_subspace, npcs_keep,
                                                demean=True, 
                                                normalization=None, plot_orthonormalization=False, 
                                                PLOT_COEFF_HEATMAP=False, savedir_coeff_heatmap=None, PRINT=False,
                                                get_axis_for_categorical_vars=True, savedir_pca_subspaces=None):
        """
        [GOOD] Get subspace for a set of variables, and then project data into that subspace.
        one_var --> returns ndim, projected to this var, where dims are defined by PCA on the basis set spanned by
        the levels for this categorical var.

        PARAMS:
        - variables, list of str, the variables that will be fit using regression, and from which the regression coefficients
        will be used to get the subspace. REgression performed independently for each neuron.
        PARAMS:
        - variables, list of str, the variables that will be fit using regression, and from which the regression coefficients
        - variables_is_cat, list of bool, whether each variable is categorical or not.
        - list_subspaces, list of tuples, each tuple is a subspace. Each subspace is a list of variables.
        - normalization, str, either None, "norm", or "orthonormalize".
        """ 

        # for subspace_tuple in list_subspaces:
        #     assert isinstance(subspace_tuple, (tuple, list))
        #     assert isinstance(subspace_tuple[0], str)

        # Input must be scalarized
        assert self.X.shape[2]==1

        # Demean 
        if demean:
            PA = self.norm_subtract_mean_each_chan()
        else:
            PA = self

        ### Collect coefficients across all neurons
        _, dfbases, res_all, original_feature_mapping = PA.regress_neuron_task_variables_all_chans(variables, variables_is_cat, PLOT_COEFF_HEATMAP, 
                                                                      PRINT=PRINT, savedir_coeff_heatmap=savedir_coeff_heatmap,
                                                                      get_axis_for_categorical_vars=get_axis_for_categorical_vars, 
                                                                      savedir_pca_subspaces=savedir_pca_subspaces)
        
        # Get basis vectors --> dfcoeff dataframe
        tmp = dfbases[dfbases["var_subspace"] == var_subspace]
        assert len(tmp)==1
        Xpca = tmp["Xpca"].values[0]
        explained_variance_ratio_ = tmp["explained_variance_ratio_"].values[0]
        dfcoeff = pd.DataFrame(Xpca, columns=range(Xpca.shape[1]))

        # Given PCA of the categorical variables, project onto any subspace
        # - keep maximum num dims
        ndims = Xpca.shape[1]
        if npcs_keep>ndims:
            npcs_keep = ndims
        # make this ndims subspace
        subspace_tuple = tuple(range(npcs_keep))
        list_subspaces = [subspace_tuple]
        dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed = PA.dataextract_subspace_targeted_pca_project(
            dfcoeff, list_subspaces, normalization, plot_orthonormalization)

        pa_subspace = dict_subspace_pa[subspace_tuple]
        subspace_axes_orig = dict_subspace_axes_orig[subspace_tuple]
        subspace_axes_normed = dict_subspace_axes_normed[subspace_tuple]

        return pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PA

    def dataextract_project_data_denoise(self, basis_vectors, version="projection", 
                                         normalization=None, plot_orthonormalization=False):
        """
        PARAMS:
        - (nchans, ndims_project), where nchans matches self.CHans
        - do_orthonormal, bool, if True, then orthonormlaizes the basis using QR decomspotion. The order of columns
        in basis matters. ie sequentially gets orthognalizes each column by the subspace spanned by the preceding columns.
        """
        from neuralmonkey.analyses.state_space_good import dimredgood_project_data_denoise_simple
        Xnew, basis_vectors = dimredgood_project_data_denoise_simple(self.X, basis_vectors, version, normalization, 
                                                                     plot_orthonormalization)
        PAredu = self.copy_replacing_X(Xnew[:, :, None])
        return PAredu, basis_vectors

    def dataextract_pca_demixed_subspace_project(self, pca, pca_twind, pca_tbindur, pca_tbin_slice, reshape_method,
                                                 dimredgood_pca_project_do_reshape, n_pcs_subspace_max, do_pca_after_project_on_subspace,
                                                 savedir_plots):
        """
        Project raw data into pc space.
        Assumes that self.X is already preprocessed to allow directly projecting with components from pca
        PARAMS:
        - pca, pc matrices. See use within.
        RETURNS:
        - Xredu, PAredu, stats_redu, Xfinal_before_redu, pca
        """
        from neuralmonkey.analyses.state_space_good import dimredgood_pca_project

        ########### Project RAW data back into this space
        # - preprocess data
        Xfinal_before_redu, PAfinal_before_redu, _, _, _ = self.dataextract_state_space_decode_flex(pca_twind, pca_tbindur, pca_tbin_slice,
                                                                                 reshape_method=reshape_method, pca_reduce=False,
                                                                                 norm_subtract_single_mean_each_chan=False)
        
        ### Project all raw data
        if savedir_plots is not None:
            plot_pca_explained_var_path = f"{savedir_plots}/expvar_reproj_raw.pdf"
        else:
            plot_pca_explained_var_path = None
        Xredu, stats_redu, Xredu_in_orig_shape = dimredgood_pca_project(pca["components"], 
                                                                        Xfinal_before_redu, 
                                                                        plot_pca_explained_var_path=plot_pca_explained_var_path,
                                                                        do_additional_reshape_from_ChTrTi=dimredgood_pca_project_do_reshape)

        if dimredgood_pca_project_do_reshape:
            # TRAJ, returns (ndims, ntrials,  ntimes)
            del Xredu

            # Keep only n final dimensions in subspace
            if n_pcs_subspace_max is not None and n_pcs_subspace_max <= Xredu_in_orig_shape.shape[0]:
                Xredu_in_orig_shape = Xredu_in_orig_shape[:n_pcs_subspace_max, :, :]

            if do_pca_after_project_on_subspace:
                assert False, "not yet coded for trajecroreis"
                # Optionally, do PCA again on raw data that was projected into this subspace (useful for visualization).
                # [IGNORE -- this is counterproductive]. This is just for rotation...
                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                plot_pca_explained_var_path_this = f"{savedir_plots}/expvar_pca_after_subspace_projection.pdf"
                plot_loadings_path_this = f"{savedir_plots}/loadings_pca_after_subspace_projection.pdf"
                Xredu, _, _ = dimredgood_pca(Xredu, how_decide_npcs_keep = "keep_all",
                                plot_pca_explained_var_path=plot_pca_explained_var_path_this,
                            plot_loadings_path=plot_loadings_path_this)

            # Get a PA holding final projected data
            PAredu = PAfinal_before_redu.copy_replacing_X(Xredu_in_orig_shape)

            # print(Xredu_in_orig_shape.shape) # (npcs, ntrials, ntimes)
            # print(PAredu.X.shape) # (npcs, ntrials, ntimes)
            # print(Xfinal_before_redu.shape) # (nchans, ntrials, ntimes)
            # print(pca["X_before_dimred"].shape) # (nchans, ngroups, ntimes)
            # assert False
            return Xredu_in_orig_shape, PAredu, stats_redu, Xfinal_before_redu, pca

        else:
            # SCALAR, returns (ntrials, ndims)

            # Keep only n final dimensions in subspace
            if n_pcs_subspace_max is not None and n_pcs_subspace_max <= Xredu.shape[1]:
                Xredu = Xredu[:, :n_pcs_subspace_max]

            if do_pca_after_project_on_subspace:
                # Optionally, do PCA again on raw data that was projected into this subspace (useful for visualization).
                # [IGNORE -- this is counterproductive]. This is just for rotation...
                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                plot_pca_explained_var_path_this = f"{savedir_plots}/expvar_pca_after_subspace_projection.pdf"
                plot_loadings_path_this = f"{savedir_plots}/loadings_pca_after_subspace_projection.pdf"
                Xredu, _, _ = dimredgood_pca(Xredu, how_decide_npcs_keep = "keep_all",
                                plot_pca_explained_var_path=plot_pca_explained_var_path_this,
                            plot_loadings_path=plot_loadings_path_this)

            # Get a PA holding final projected data
            assert len(PAfinal_before_redu.Xlabels["trials"]) == Xredu.shape[0]
            PAredu = PopAnal(Xredu.T[:, :, None].copy(), [0])  # (ndimskeep, ntrials, 1)
            PAredu.Xlabels = {dim:df.copy() for dim, df in PAfinal_before_redu.Xlabels.items()}

            # print(Xredu.shape)
            # print(PAredu.X.shape)
            # print(Xfinal_before_redu.shape)
            # print(pca["X_before_dimred"].shape)
            # assert False
            return Xredu, PAredu, stats_redu, Xfinal_before_redu, pca


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
        - twind_overall, only keep data within this window (e.g, [0.3, 0.6]). None, to keep all data
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
        elif tbin_dur=="ignore":
            # Pass, no smoothing
            pass
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
        if reshape_method == "chans_x_trialstimes":
            # E.g. if decoding, soemtimes want each time bin as datapt.

            dflab = PAslice.Xlabels["trials"]
            list_x = []
            list_dflab = []
            # labels_all = []
            for i in range(ntimes):
                list_x.append(PAslice.X[:, :, i])
                # labels_all.extend(labels)
                list_dflab.append(dflab)
            X_before_dimred = np.concatenate(list_x, axis=1) # (nchans, ntrials x ntimes)
            dflab_final = pd.concat(list_dflab).reset_index(drop=True)

            if pca_reduce:  
                assert False, "not codede yet"
            else:
                X = X_before_dimred
                pca = None

            assert extra_dimred_method is None, "Not codede yet"

            # Represent X in PopAnal
            PAfinal = PopAnal(X[:, :, None], times=[0], chans=PAslice.Chans)  # (ndimskeep, ntrials x ntimes, 1)
            PAfinal.Xlabels = {dim:df.copy() for dim, df in PAslice.Xlabels.items()}
            PAfinal.Xlabels["trials"] = dflab_final
            assert len(PAfinal.Xlabels["trials"])==PAfinal.X.shape[1]

            # # Sanity check
            # assert X.shape[0] == PAslice.X.shape[1]
            # assert X.shape[1] == PAfinal.X.shape[0]
            # assert X.shape[0] == PAfinal.X.shape[1]

        elif reshape_method=="trials_x_chanstimes":
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
            # Then reshapes back ot (ndims, ntrials, ntimes)

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

            ################## Get in shape.
            X = np.reshape(PAslice.X, [nchans, ntrials * ntimes]).T # (ntrials*ntimes, nchans)
            
            if pca_reduce:
                print("Doing PCA")
                # Reshape to pass into PCA

                # print(X.shape)
                # print(np.mean(PAslice.X[0, 0]))
                # print("HEREsadasd", np.mean(X, axis=0))
                # assert False
                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                # Make labels (chans x timebins)
                ntimes = len(PAslice.Times)
                col_labels = PAslice.Chans
                # col_labels = []
                # for ch in PAslice.Chans:
                #     for t in range(ntimes):
                #         col_labels.append((ch, t))
                X, _, pca = dimredgood_pca(X,
                                           how_decide_npcs_keep = how_decide_npcs_keep,
                                           pca_frac_var_keep=pca_frac_var_keep, pca_frac_min_keep=pca_frac_min_keep,
                                           plot_pca_explained_var_path=plot_pca_explained_var_path,
                                           plot_loadings_path=plot_loadings_path,
                                           plot_loadings_feature_labels=col_labels,
                                           method=pca_method,
                                           npcs_keep_force=npcs_keep_force) # (ntrials, nchans) --> (ntrials, ndims)
            # Extra dimreduction step?
            if extra_dimred_method in ["umap", "mds"]:
                from neuralmonkey.analyses.state_space_good import dimredgood_nonlinear_embed_data
                # print(X.shape)
                # print(extra_dimred_method)
                # print(extra_dimred_method_n_components)
                # print(umap_n_neighbors)
                # assert False
                X, _ = dimredgood_nonlinear_embed_data(X, METHOD=extra_dimred_method, n_components=extra_dimred_method_n_components,
                                                           umap_n_neighbors=umap_n_neighbors) # 
            else:
                assert extra_dimred_method is None


            ################# Reshape back to original
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

            # # Extra dimreduction step?
            # assert extra_dimred_method is None, "not yet coded.. a bit tricky?"

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


    ###################### EUCLIDIAN DISTNACE
        
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


    def slice_by_label_grouping(self, grouping_variables):
        """ 
        Return sliced PA, where first constructs grouping varible (can be conjunctive)
        then only keep desired subsets 
        PARAMS;
        - dim_str, e.g,., "trials"
        - grouping_variables, list of str
        - grouping_values, values to keep
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        dflab = self.Xlabels["trials"]
        grpdict = grouping_append_and_return_inner_items_good(dflab, grouping_variables)
        pa_dict = {}
        for grp, inds in grpdict.items():
            pa = self.slice_by_dim_indices_wrapper("trials", inds, reset_trial_indices=True)
            pa_dict[grp] = pa
        
        return pa_dict

    def slice_and_agg_wrapper(self, along_dim, grouping_variables, grouping_values=None,
            agg_method = "mean", return_group_dict=False, return_list_pa=False, min_n_trials_in_lev=1):
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
        groupdict_keep = {}
        for grp in groupdict:
            inds = groupdict[grp]

            # print(grp, len(inds))
            if len(inds)>=min_n_trials_in_lev:
                # slice
                pathis = self.slice_by_dim_indices_wrapper(dim=along_dim, inds=inds)

                # agg
                pathis = pathis.agg_wrapper(along_dim=along_dim, agg_method=agg_method)

                # collect
                list_pa.append(pathis)
                list_grplevel.append(grp)
                groupdict_keep[grp] = inds
        groupdict = groupdict_keep

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


    ################ BEHAVIOR
    def behavior_extract_neural_stroke_aligned(self, ind_trial, lag_neural_vs_beh=0., 
                                               var_strok="strok_beh", PRINT=False, PLOT=False):
        """
        Extract neural and stroke data, where neural data is clipped to align to stroke.

        PARAMS:
        - lag_neural_vs_beh = 0. # if negative this means get neural data that precedes behavior.

        NOTE: assumes that self is stroke data -- ie. that time 0 is aligned to stroke onset.
        NOTE: Deals with cases where there is not neough neural data to match duration of stroke, by
        clipping off the end of the stroke, to match whatever neural data exists.

        MINOR PROBLEM -- this runs over all trial sfor pa, but aligned to a single trail for beh... is
        inefficient.
        """
        from pythonlib.tools.stroketools import strokesInterpolate2

        pa_timestamp_stroke_onset = 0. # assumes self is stroke data.

        self.behavior_extract_strokes_to_dflab(trial_take_first_stroke=True)
        dflab = self.Xlabels["trials"]
        strokes = dflab[var_strok].tolist()
        # Condition the stroke
        strok = strokes[ind_trial]

        # Get strok, with times now zeroed to stroke onset
        strok = strok.copy()
        strok[:, 2] -= strok[0, 2] # to zero them
        dur_take = strok[-1, 2] - strok[0, 2] # how long is the stroke?

        if PRINT:
            print("Starting times:")
            print(self.Times)
            print(strok[:, 2])
            print(strok[-1, 2] - strok[0, 2])

        # pull out time window from neural data matching the stroke
        t1 = pa_timestamp_stroke_onset + lag_neural_vs_beh
        t2 = pa_timestamp_stroke_onset + dur_take + lag_neural_vs_beh

        if PLOT:
            fig, axes = plt.subplots(2,1, sharex=True, sharey=True)
            ax = axes.flatten()[0]
            ax.plot(strok[:,2], strok[:,0], "b-", label="stroke")
            ax.plot(self.Times, self.X[0, ind_trial, :], "r-", label="neural")
            ax.set_title("before align")
            ax.legend()
            ax.axvline(t1, color="r")
            ax.axvline(t2, color="r")

        ###
        pa = self.slice_by_dim_indices_wrapper("trials", [ind_trial]).slice_by_dim_values_wrapper("times", [t1, t2])
        times_to_get_beh = pa.Times - lag_neural_vs_beh # undo the lag -- these are the times of beh that are desired
        assert np.all(times_to_get_beh>=0)

        # If not enough neural data, then need to clip off end of storkes
        strok = strok[strok[:, 2]<=times_to_get_beh[-1]+0.01, :]

        # Interpolate the beh so that it matches the neural time bins.
        strok = strokesInterpolate2([strok], ["input_times", times_to_get_beh], plot_outcome=False)[0]
        assert np.all(strok[:, 2] == times_to_get_beh)

        if PLOT:
            ax = axes.flatten()[1]
            ax.plot(strok[:,2], strok[:,0], "b-", label="stroke")
            ax.plot(pa.Times, pa.X[0, 0, :], "r-", label="neural")
            ax.set_title("after align")
            ax.set_xlabel("time (rel stroke onset)")
            ax.legend()

        if PRINT:
            print(len(pa.Times))
            print(strok.shape)
            print("neural times: ", pa.Times)
            print("beh times: ", times_to_get_beh)
            print("beh times: ", strok[:,2])

        return pa, strok
    
    def behavior_strokes_kinematics_stats(self, trial_take_first_stroke=True):
        """
        Extract strokes, and get variuos stats related to kinmetaics.
        """
        
        assert trial_take_first_stroke, "assumes one stroke below.."

        self.behavior_extract_strokes_to_dflab()
        dflab = self.Xlabels["trials"]
        strokes = dflab["strok_beh"].tolist()

        ### MOTOR PARAMETERS
        # get the initial velocity (angle and magnitude)
        from pythonlib.dataset.dataset_strokes import DatStrokes
        DS = DatStrokes()
        
        ### For each stroke, compute some features
        import numpy as np
        from pythonlib.tools.stroketools import sliceStrokes, slice_strok_by_frac_bounds
        from pythonlib.tools.stroketools import feature_velocity_vector_angle_norm

        # (1) For initial angle, take start of each stroke 
        twind = [0, 0.15] # sec
        strokes_sliced = sliceStrokes(strokes, twind, time_is_relative_each_onset=True, assert_no_lost_strokes=True)

        # - velocity vector
        angles = [] 
        norms = []
        for strok in strokes_sliced:
            _, a, n = feature_velocity_vector_angle_norm(strok)
            angles.append(a)
            norms.append(n)
        
        # - circularity
        from pythonlib.drawmodel.features import strokeCircularity
        fraclow = 0
        frachigh = 0.5
        strokes_sliced = [slice_strok_by_frac_bounds(s, fraclow, frachigh) for s in strokes]
        # strokes_sliced = slice_strok_by_frac_bounds(strokes, twind, time_is_relative_each_onset=True, assert_no_lost_strokes=True)

        # circularities = strokeCircularity(strokes)
        circularities = strokeCircularity(strokes_sliced)

        # - location of onset.
        # (reach angle, relative to on location)
        onsets_x = [strok[0, 0] for strok in strokes]
        onsets_y = [strok[0, 1] for strok in strokes]

        # Bin the angles
        from pythonlib.tools.vectools import bin_angle_by_direction
        angles_binned = bin_angle_by_direction(angles, num_angle_bins=8)

        # Put motor variables back into dflab
        dflab["motor_angle"] = angles
        # dflab["motor_angle_sin"] = angles

        dflab["motor_angle_binned"] = angles_binned

        if "gap_from_prev_angle" in dflab:
            dflab["gap_from_prev_angle_binned"] = bin_angle_by_direction(dflab["gap_from_prev_angle"].values, num_angle_bins=8)

        dflab["motor_norm"] = norms

        dflab["motor_circ"] = circularities

        dflab["motor_onsetx"] = onsets_x

        dflab["motor_onsety"] = onsets_y

        self.Xlabels["trials"] = dflab

    def behavior_extract_strokes_to_dflab(self, trial_take_first_stroke=False):
        """
        Extracts strokes_beh and strokes_task to dflab
        """
        dflab = self.Xlabels["trials"]

        if ("strok_beh" in dflab) and ("strok_task" in dflab):
            assert trial_take_first_stroke, "hacky only workse for this, assumes it is this"
            # if trial_take_first_stroke:
            #     # Then take the first stroke
            #     for col in ["strok_beh", "strok_task"]:
            #         strokes = dflab[col].tolist()
            #         strokes = [s[0] for s in strokes]
            #         dflab[col] = strokes
            return

        # Collect all the strokes
        strokes_task = []
        strokes_beh = []
        if "Tkbeh_stkbeh" in dflab.columns:
            # Then this is "trial" version
            for i, row in dflab.iterrows():
                
                tokens = row["Tkbeh_stkbeh"].Tokens
                if not trial_take_first_stroke:
                    assert len(tokens)==1, "hacky, currneyl only workse for single prim tasks"
                strok_beh = tokens[0]["Prim"].Stroke()

                tokens = row["Tkbeh_stktask"].Tokens
                if not trial_take_first_stroke:
                    assert len(tokens)==1, "hacky, currneyl only workse for single prim tasks"
                strok_task = tokens[0]["Prim"].Stroke()

                strokes_task.append(strok_task)
                strokes_beh.append(strok_beh)
        elif "Stroke" in dflab.columns:
            # Then this is "stroke" version
            for i, row in dflab.iterrows():
                
                strok_beh = row["Stroke"]()
                
                tokens = row["TokTask"].Tokens
                assert len(tokens)==1, "hacky, currneyl only workse for single prim tasks"
                strok_task = tokens[0]["Prim"].Stroke()

                strokes_task.append(strok_task)
                strokes_beh.append(strok_beh)
        else:
            print(sorted(dflab.columns))
            assert False, "where are strokes saved?"

        dflab["strok_beh"] = strokes_beh
        dflab["strok_task"] = strokes_task

    def behavior_replace_neural_with_strokes(self, version="beh", n_time_bins=50,
                                             centerize_strokes=True, plot_examples=False,
                                             remove_time_axis=True,
                                             align_strokes_to_onset=False):
        """
        Replace self.X with strokes, such that shape is now (2, ntrials, ntimes), where
        the 2 are x and y.
        PARAMS:
        - n_time_bins, for interpolation
        RETURNS:
        - PAstroke, copy of self, with X replaced, shape (2 or 3, ntrials, n_time_bins), where
        2 or 3 depends on if remove_time_axis.
        """
        from pythonlib.tools.stroketools import strokesInterpolate2, strokes_centerize
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper

        # Extract beh and task strokes
        self.behavior_extract_strokes_to_dflab()

        # interpolate so that all strokes are same length
        dflab = self.Xlabels["trials"]
        if version == "beh":
            strokes = dflab["strok_beh"].tolist()
        else:
            strokes = dflab["strok_task"].tolist()
        strokes = strokesInterpolate2(strokes, ["npts", n_time_bins], base="time", plot_outcome=False)

        # center each stroke.
        if centerize_strokes:
            strokes = strokes_centerize(strokes)

        if align_strokes_to_onset:
            from pythonlib.tools.stroketools import strokes_alignonset
            strokes = strokes_alignonset(strokes)

        if plot_examples:
            fig, ax = plt.subplots()
            plotDatStrokesWrapper(strokes[:4], ax)
        
        X = np.stack(strokes, axis=0) # (ntrials, ntimes, ndims)
        X = np.transpose(X, (2, 0, 1)) # (ndims, ntrials, ntimes)

        if remove_time_axis:
            # remove the time axis
            X = X[:2, :, :]

        # replace PAredu
        PAstroke = self.copy_replacing_X(X)

        return PAstroke
    
    def behavior_extract_events_timing(self, MS, events=None, normalize_to_this_event=None):
        """
        Extract times of events for each trial, using the original MS (sessions).
        Optimized for "stroke"-level data.
        PARAMS:
        - normalize_to_this_event, str, if not None, then each event time subtracts this time.
        RETURNS:
        - dftimes, each row matches correspnding row of self.Xlabels["trials"], and columns hold events
        - events, list of str.
        """

        dflab = self.Xlabels["trials"]

        if events is None:
            # currently optimized for strokes data
            events = ["go", "first_raise", "on_strokeidx_0", "off_strokeidx_0"]

        list_inds = list(range(len(dflab)))
                         
        res = []
        for ind in list_inds:
            tc = dflab.iloc[ind]["trialcode"]

            sn, trial_sn, _ = MS.index_convert_trial_trialcode_flex(tc)

            # Get times of events for this trials
            event_times = {}
            for ev in events:
                event_times[ev] = sn.events_get_time_helper(ev, trial_sn, assert_one=True)[0]

            res.append(event_times)
            res[-1]["trialcode"] = tc
            res[-1]["ind_dflab"] = ind


            # try:
            if dflab.iloc[ind]["event"] == "00_stroke" and dflab.iloc[ind]["stroke_index"] == 0 and "on_strokeidx_0" in events:
                # Then do sanity check that times match...
                assert np.abs(dflab.iloc[ind]["event_time"] - event_times["on_strokeidx_0"]) < 0.05, "touchscreen lag? this is arleayd accounted for with 0.05"
            # except Exception as err:
            #     print(event_times["on_strokeidx_0"])
            #     print(dflab.iloc[ind]["event_time"])
            #     print(dflab.iloc[ind])
            #     raise err

        dftimes = pd.DataFrame(res)

        # Normalie buy subtracting stroke onset time
        if normalize_to_this_event is not None:
            tmp = dftimes[normalize_to_this_event].copy()
            for ev in events:
                dftimes[ev] = dftimes[ev] - tmp
        
        return dftimes, events

    def behavior_extract_events_timing_plot_distributions(self, MS, shape_var, events=None, 
                                                          normalize_to_this_event=None, xlims=None):
        """
        [pretty specific] plot timing of events surrounding stroke onset.
        Tailored for strokes-data.
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        from pythonlib.tools.plottools import legend_add_manual

        # Extract timing of events
        dftimes, events = self.behavior_extract_events_timing(MS, events=events, 
                                                              normalize_to_this_event=normalize_to_this_event)
        
        # Prep plot
        dflab = self.Xlabels["trials"]
        map_event_to_color, _, _= color_make_map_discrete_labels(events)
        grpdict = grouping_append_and_return_inner_items_good(dflab, [shape_var])
        SIZE = 2
        ncols = 6
        n = len(grpdict)+1
        nrows = int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE*2, nrows*SIZE), sharex=True, sharey=True)

        for ax, (grp, inds) in zip(axes.flatten(), grpdict.items()):

            dfthis = dftimes.iloc[inds]
            for ev in events:
                col = map_event_to_color[ev]
                times = dfthis[ev].values
                ax.plot(times, np.ones(len(times))+0.1*(np.random.rand(len(times))-0.5), ".", 
                        label=ev, alpha=0.35, color=col)
            ax.set_ylim([0.5, 1.5])

            ax.set_xlabel("time (sec)")
            ax.set_title(grp)

            if xlims is not None:
                ax.set_xlim(xlims)

        ax = axes.flatten()[-1]
        legend_add_manual(ax, map_event_to_color.keys(), map_event_to_color.values())

        return fig

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
        if twind[0] > max(self.Times) or twind[1] < min(self.Times):
            print("-----")
            print(twind)
            print(self.Times)
            print(inds)
            assert False, "twind incompatible with times"
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

    def shuffle_by_time(self):
        """ 
        Return a copy of self.X, with time dimension shuffled,
        i.e., shape is identical.
        """

        # get random indices for shuffling
        n = self.X.shape[2]
        inds = np.random.permutation(range(n))
        X = self.X[:, :, inds].copy()
        return X

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
        return plotNeurHeat(X, times=self.Times, **kwargs)

    def plotNeurTimecourse(self, trial, version="raw", **kwargs):
        X = self.extract_activity_copy(trial, version)
        return plotNeurTimecourse(X, times=self.Times, **kwargs)

    def plotwrapper_smoothed_fr(self, values_this_axis=None, axis_for_inds="chans", ax=None, 
                     plot_indiv=True, plot_summary=False, error_ver="sem",
                     pcol_indiv = "k", pcol_summary="r", summary_method="mean",
                     event_bounds=(None, None, None), alpha=0.6, 
                     time_shift_dur=None, plot_indiv_n_rand = 10):
        """ [GOOD: low-level] Wrapper for different ways of plotting multiple smoothed fr traces, where
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
        from neuralmonkey.neuralplots.population import plotNeurTimecourse, plot_smoothed_fr, plotNeurTimecourseErrorbar
        import numpy as np

        # extract data
        if values_this_axis is not None:
            PAthis = self.slice_by_dim_values_wrapper(axis_for_inds, values_this_axis)
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
        
        if False:
            if X.shape[1]==1 and axis_for_inds not in ["trials"]:
                plot_indiv = True
                plot_summary = False

        if time_shift_dur is not None:
            times = times + time_shift_dur
            
        # Reshape to (nsamp, times), combining all chans and trials.
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        n_time_bins = X.shape[1]

        # 1) Plot indiividual traces?
        if plot_summary==True and isinstance(pcol_indiv, str) and pcol_indiv=="k":
            # Then this is default pcol color. make it the same as the sumary color.
            pcol_indiv = pcol_summary

        if plot_indiv:
            fig1, ax1 = plotNeurTimecourse(X, times, ax=ax, color = pcol_indiv,
                alpha=alpha, n_rand=plot_indiv_n_rand)
        else:
            fig1, ax1 = None, None

        # 2) Plot summary too?
        if plot_summary:
            fig2, ax2 = plot_smoothed_fr(X, times, ax, summary_method=summary_method,
                             color=pcol_summary)
            # if error_ver=="sem":
            #     from scipy import stats
            #     if summary_method=="mean":
            #         Xmean = np.mean(X, axis=0)
            #     elif summary_method=="median":
            #         Xmean = np.median(X, axis=0)
            #     else:
            #         print(summary_method)
            #         assert False
            #     Xsem = stats.sem(X, axis=0)
            # else:
            #     assert error_ver is None, "not coded"

            # fig2, ax2 = plotNeurTimecourseErrorbar(Xmean, Xerror=Xsem, times=times,ax=ax, color=pcol_summary)
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
                                              event_bounds=(None, None, None),
                                              add_legend=True, legend_levels=None,
                                               chan=None, dict_lev_color=None,
                                               plot_indiv_n_rand=10):
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
            # assert False, "did you really not want to input chan?"
            PA = self

        if ax is None:
            fig, ax = plt.subplots()
            
        # Split into each pa for each level
        list_pa, list_levels_matching_pa = PA.split_by_label(dim_str, dim_variable)
        
        # make dict mapping from level to col
        if legend_levels is None:
            # then use the levels within here
            legend_levels = list_levels_matching_pa

        if dict_lev_color is None:
            from pythonlib.tools.plottools import color_make_map_discrete_labels
            dict_lev_color, _, _ = color_make_map_discrete_labels(list_levels_matching_pa)

        # dict_lev_color = {}
        # pcols = makeColors(len(legend_levels))
        # for pc, lev in zip(pcols, legend_levels):
        #     dict_lev_color[lev] = pc

        for pa, lev in zip(list_pa, list_levels_matching_pa):
            pcol = dict_lev_color[lev]
            pa.plotwrapper_smoothed_fr(ax=ax, plot_indiv=plot_indiv, plot_summary=plot_summary,
                                          pcol_indiv = pcol, pcol_summary=pcol,
                                          event_bounds=event_bounds, plot_indiv_n_rand=plot_indiv_n_rand)
        # add legend
        if add_legend:
            from pythonlib.tools.plottools import legend_add_manual
            legend_add_manual(ax, dict_lev_color.keys(), dict_lev_color.values(), 0.2)

        return dict_lev_color.values()

    def plotwrappergrid_smoothed_fr_splot_var(self, var_row, var_col, chans, 
                                              plot_indiv=False, do_sort=True,
                                              plot_indiv_n_rand=10):
        """
        Smoothed FR, in a 
        Grid of subplots:
        Subplot = (var_row, var_col); Colors = dimensions/chans

        EXAMPLE:
        var_col = "seqc_0_shape", or list of col
        var_row = "seqc_0_loc"
        chans = [0,1,2,3,4]
        """

        # Prune to just the chans that exist
        chans = [ch for ch in chans if ch in self.Chans]

        if isinstance(var_col, (list, tuple)):
            from pythonlib.tools.pandastools import append_col_with_grp_index
            self.Xlabels["trials"] = append_col_with_grp_index(self.Xlabels["trials"], var_col, "_var_col")
            var_col = "_var_col"

        grpvars = [var_col, var_row]
        pa_dict = self.slice_by_label_grouping(grpvars)

        levels_col = list(set([x[0] for x in pa_dict.keys()]))
        levels_row = list(set([x[1] for x in pa_dict.keys()]))
        if do_sort:
            levels_col = sort_mixed_type(levels_col)
            levels_row = sort_mixed_type(levels_row)

        ncols = len(levels_col)
        nrows = len(levels_row)

        from pythonlib.tools.plottools import color_make_map_discrete_labels
        map_chan_to_color, _, _ = color_make_map_discrete_labels(chans)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), sharex=True, sharey=True, squeeze=False)
        for i, levcol in enumerate(levels_col):
            for j, levrow in enumerate(levels_row):
                ax = axes[j][i]
                if (levcol, levrow) in pa_dict.keys():
                    pa = pa_dict[(levcol, levrow)]

                    for ch in chans:
                        pcol = map_chan_to_color[ch]
                        pa.plotwrapper_smoothed_fr([ch], axis_for_inds="chans", ax=ax, plot_indiv=plot_indiv, plot_summary=True,
                                                pcol_summary=pcol, pcol_indiv = "k", plot_indiv_n_rand=plot_indiv_n_rand)

                    ax.set_title((levcol, levrow))
                else:
                    ax.set_title("missing data")

                if i==len(levels_col)-1 and j==len(levels_row)-1:
                    from pythonlib.tools.plottools import legend_add_manual
                    legend_add_manual(ax, map_chan_to_color.keys(), map_chan_to_color.values(), 0.2)
        return fig

    def plotwrappergrid_smoothed_fr_splot_neuron(self, var_effect, vars_others, chans, plot_indiv=False,
                                                 plot_indiv_n_rand=10, plot_summary=True, sharey=True):
        """
        Grid of subplots...
        Subplot = (chan, vars_others); Colors = var_effect
        PARAMS:
        - chans, list of values in self.Chans (not the indices)
        EXAMPLE:
        # vars_subplots = ["seqc_0_loc"]
        # var = "seqc_0_shape"
        # chans = [0,1000, 2]

        NOTE: This replaces 
        """

        # Prune to just the chans that exist
        chans_orig = [ch for ch in chans]
        chans = [ch for ch in chans if ch in self.Chans]

        if len(chans)==0:
            print("chans you want to plot:", chans_orig)
            print("self.Chans:", self.Chans)

        list_pa, levels = self.split_by_label("trials", vars_others)
        ncols = len(levels)
        nrows = len(chans)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), sharex=True, sharey=sharey, squeeze=False)
        for i, (lev, pa) in enumerate(zip(levels, list_pa)):
            for j, ch in enumerate(chans):
                ax = axes[j][i]

                add_legend = (i==len(levels)-1) and (j==len(chans)-1)
                pa.plotwrapper_smoothed_fr_split_by_label("trials", var_effect, ax, chan=ch, add_legend=add_legend,
                    plot_indiv=plot_indiv, plot_indiv_n_rand=plot_indiv_n_rand, plot_summary=plot_summary)
                ax.set_title(lev)

                if i==0:
                    ax.set_ylabel(f"ch {ch}")

        if sharey==False:
            # Make sure y is still shared within each row (each channel).
            from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
            share_axes_row_or_col_of_subplots(axes, "row", "y")

        return fig

    def plotwrapper_smoothed_fr_split_by_label_and_subplots(self, chan, var, vars_subplots):
        """
        Helper to plot smoothed fr, multkiple supblots, each varying by var
        :param var: str, to splot and color within subplot
        :param vars_subplots: list of str, each is a supblot
        :param chan: value in self.Chans
        :return:
        """
        list_pa, levels = self.split_by_label("trials", vars_subplots)
        ncols = min([len(list_pa), 8])
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



    def plot_state_space_good_wrapper(self, savedir, LIST_VAR, LIST_VARS_OTHERS=None, LIST_FILTDICT=None, LIST_PRUNE_MIN_N_LEVS=None,
                                      time_bin_size = 0.05, PLOT_CLEAN_VERSION = False, 
                                      nmin_trials_per_lev=None, list_dim_timecourse=None, list_dims=None,
                                      also_plot_heatmaps=False):
        """
        Wrapper to make ALL state space plots, including (i) trajectiroeis (ii) scalars, and (iii) traj vs. time plots.
        PARAMS:
        - LIST_VAR, list of str, variable to use for coloring plots. Makes separate plots.
        - LIST_VARS_OTHERS, list of list/tuples of strings. for splitting into subplots.
        - LIST_FILTDICT, for filtering before plotting.
        - LIST_PRUNE_MIN_N_LEVS, list of int.
        - PLOT_CLEAN_VERSION, bool, if true, then amkes the plot prety.
        - nmin_trials_per_lev, prunes levels with fewer trials thant his.
        - time_bin_size, this is the final separation in time between the pts that are plotted (i.e., is "slide")
        """

        from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER, trajgood_plot_colorby_splotby_WRAPPER
        from neuralmonkey.analyses.state_space_good import trajgood_construct_df_from_raw, trajgood_plot_colorby_splotby_timeseries

        PAorig = self.copy()
        if LIST_VARS_OTHERS is None:
            PAorig.Xlabels["trials"]["dummy"] = "dummy"
            LIST_VARS_OTHERS = [["dummy"] for _ in range(len(LIST_VAR))]
        if LIST_FILTDICT is None:
            LIST_FILTDICT = [None for _ in range(len(LIST_VAR))]
        if LIST_PRUNE_MIN_N_LEVS is None:
            LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
            
        assert len(LIST_VARS_OTHERS) == len(LIST_VAR)
        assert len(LIST_FILTDICT) == len(LIST_VAR)
        assert len(LIST_PRUNE_MIN_N_LEVS) == len(LIST_VAR)

        if list_dims is None:
            if len(LIST_VAR)<15:
                if self.X.shape[0]==3:
                    list_dims = [(0,1), (1,2)]
                elif self.X.shape[0]>3:
                    list_dims = [(0,1), (2,3)]
                else:
                    list_dims = [(0,1)]
            else:
                # Too slow, just do 1st 2 d
                list_dims = [(0,1)]

        list_dims = [dims for dims in list_dims if max(dims)<self.X.shape[0]]

        if list_dim_timecourse is None:
            list_dim_timecourse = [0, 1]

        list_dim_timecourse = [dim for dim in list_dim_timecourse if dim<self.X.shape[0]]

        ### Plot each set of var, var_others
        vars_already_state_space_plotted = []
        var_varothers_already_plotted = []
        heatmaps_already_plotted = []
        for i_var, (var, var_others, filtdict, prune_min_n_levs) in enumerate(zip(LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS)):
            print("RUNNING: ", i_var,  var, " -- ", var_others)

            # Copy pa for this
            PA = PAorig.copy()

            ####################### Cleanup PA
            var_for_name = var
            if isinstance(var, (tuple, list)):
                PA.Xlabels["trials"] = append_col_with_grp_index(PA.Xlabels["trials"], var, "_tmp")
                var = "_tmp"

            if filtdict is not None:
                for _var, _levs in filtdict.items():
                    print("len pa bnefore filt this values (var, levs): ", _var, _levs)
                    PA = PA.slice_by_labels("trials", _var, _levs, verbose=True)

            if nmin_trials_per_lev is not None:
                prune_min_n_trials = nmin_trials_per_lev
            else:
                prune_min_n_trials = 5

            if (var, tuple(var_others)) not in heatmaps_already_plotted:
                plot_counts_heatmap_savepath = f"{savedir}/{i_var}_counts_heatmap-var={var_for_name}-ovar={'|'.join(var_others)}.pdf"
                heatmaps_already_plotted.append((var, tuple(var_others)))
            else:
                plot_counts_heatmap_savepath = None

            PA, _, _= PA.slice_extract_with_levels_of_conjunction_vars(var, var_others, prune_min_n_trials, prune_min_n_levs,
                                                            plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
            if PA is None:
                print("all data pruned!!")
                continue
            
            # DEcide if this is trajectory or scalar.
            if PA.X.shape[2]>1:
                # Then is trajecotry
                PA_traj = PA
                PA_scal = PA.agg_wrapper("times")
            else:
                PA_traj = None
                PA_scal = PA

            ### (1) var -- split by subplotvar
            if not var_others == "dummy":
                if (var, var_others) not in var_varothers_already_plotted:
                    var_varothers_already_plotted.append((var, tuple(var_others)))
                    # (1a) Traj
                    sdir = f"{savedir}/TRAJ"
                    os.makedirs(sdir, exist_ok=True)
                    if PA_traj is not None:    
                        if PLOT_CLEAN_VERSION == False:
                            trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                                sdir, var_others, list_dims, 
                                                                time_bin_size=time_bin_size, save_suffix=i_var)
                        else:
                            # Plot a "clean" version (Paper version), including with different x and y lims, so can compare
                            # across plots
                            ssuff = f"{i_var}"
                            trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                                sdir, var_others, list_dims, 
                                                                time_bin_size=None, save_suffix=ssuff,
                                                                plot_dots_on_traj=False)
                            
                            for xlim_force in [
                                # [-3.2, 3.2],
                                [-2.4, 2.4],
                                ]:
                                for ylim_force in [
                                    # [-1.5, 1.5],
                                    [-2, 2],
                                    # [-2.5, 2.5],
                                    ]:
                                    ssuff = f"{i_var}--xylim={xlim_force}|{ylim_force}"
                                    trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                                        sdir, var_others, list_dims, 
                                                                        time_bin_size=None, save_suffix=ssuff,
                                                                        plot_dots_on_traj=False,
                                                                        xlim_force = xlim_force, ylim_force=ylim_force)
                    # (1b) Scal
                    dflab = PA_scal.Xlabels["trials"]
                    Xthis = PA_scal.X.squeeze(axis=2).T # (n4trials, ndims)
                    sdir = f"{savedir}/SCALAR"
                    os.makedirs(sdir, exist_ok=True)
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, sdir,
                                                                    vars_subplot=var_others, list_dims=list_dims,
                                                                    skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                    var_varothers_already_plotted.append((var, tuple(var_others)))
                    plt.close("all")
                    

                    # (1c) Timecourse
                    if PA_traj is not None:
                        sdir = f"{savedir}/TIMECOURSE"
                        os.makedirs(sdir, exist_ok=True)

                        plot_trials_n = 5
                        if var_others is not None:
                            _vars = [var] + list(var_others)
                        else:
                            _vars = [var]

                        if False:
                            df = trajgood_construct_df_from_raw(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], _vars)
                            
                            for dim in list_dim_timecourse:
                                # - (i) combined, plotting means.
                                fig, _ = trajgood_plot_colorby_splotby_timeseries(df, var, var_others, dim=dim,
                                                                                plot_trials_n=plot_trials_n, 
                                                                                SUBPLOT_OPTION="split_levs")
                                path = f"{sdir}/TIMECOURSEsplit-color={var}-sub={var_others}-dim={dim}-suff={i_var}.pdf"
                                print("Saving ... ", path)
                                savefig(fig, path)

                                # - (2) split
                                fig, _ = trajgood_plot_colorby_splotby_timeseries(df, var, var_others, dim=dim, plot_trials_n=plot_trials_n,
                                                                        plot_trials=False, SUBPLOT_OPTION="combine_levs")
                                path = f"{sdir}/TIMECOURSEcomb-color={var}-sub={var_others}-dim={dim}-suff={i_var}.pdf"
                                print("Saving ... ", path)
                                savefig(fig, path)
                                
                                plt.close("all")
                        else:
                            try:
                                fig = PA_traj.plotwrappergrid_smoothed_fr_splot_var(var, var_others, list_dim_timecourse)
                                path = f"{sdir}/timecourse_splot_var-var={var}-varother={var_others}.pdf"
                                print("Saving ... ", path)
                                savefig(fig, path)

                                fig = PA_traj.plotwrappergrid_smoothed_fr_splot_neuron(var, var_others, list_dim_timecourse)
                                path = f"{sdir}/timecourse_splot_neur-var={var}-varother={var_others}.pdf"
                                print("Saving ... ", path)
                                savefig(fig, path)
                            except Exception as err:
                                print(PA_traj.X.shape, PA_traj.Chans)
                                print(var, var_others, list_dim_timecourse)
                                raise err


            ### (2) var  (combining across subplot vars)
            if var not in vars_already_state_space_plotted:
                vars_already_state_space_plotted.append(var)
                # (2a) Traj
                if PA_traj is not None:    
                    sdir = f"{savedir}/TRAJ"
                    os.makedirs(sdir, exist_ok=True)
                    if PLOT_CLEAN_VERSION:
                        # Plot a "clean" version (Paper version), including with different x and y lims, so can compare
                        # across plots
                        trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                            sdir, None, list_dims, 
                                                            time_bin_size=None, save_suffix=i_var,
                                                            plot_dots_on_traj=False)

                        for xlim_force in [
                            # [-3.2, 3.2],
                            [-2.4, 2.4],
                            ]:
                            for ylim_force in [
                                # [-1.5, 1.5],
                                [-2, 2],
                                # [-2.5, 2.5],
                                ]:
                                ssuff = f"{i_var}--xylim={xlim_force}|{ylim_force}"
                                trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                                    sdir, None, list_dims, 
                                                                    time_bin_size=None, save_suffix=ssuff,
                                                                    plot_dots_on_traj=False,
                                                                    xlim_force = xlim_force, ylim_force=ylim_force)
                    else:
                        trajgood_plot_colorby_splotby_WRAPPER(PA_traj.X, PA_traj.Times, PA_traj.Xlabels["trials"], var, 
                                                            sdir, None, list_dims, 
                                                            time_bin_size=time_bin_size, save_suffix=i_var)
                # (2b) Scal
                dflab = PA_scal.Xlabels["trials"]
                Xthis = PA_scal.X.squeeze(axis=2).T # (n4trials, ndims)
                sdir = f"{savedir}/SCALAR"
                os.makedirs(sdir, exist_ok=True)
                trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, sdir,
                                                                vars_subplot=None, list_dims=list_dims,
                                                                skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                var_varothers_already_plotted.append((var, tuple(var_others)))

            ########### HEATMAPS of activity
            if PA_traj is not None and also_plot_heatmaps:
                savedir_this = f"{savedir}/heatmaps-var={var}-varother={var_others}"
                os.makedirs(savedir_this, exist_ok=True)
                from neuralmonkey.neuralplots.population import heatmapwrapper_many_useful_plots
                zlims = None
                heatmapwrapper_many_useful_plots(PA_traj, savedir_this, var, var_others, False, False, zlims)

            ####
            plt.close("all")    


    ##########################
    def regress_neuron_task_variables_all_chans_plot_coeffs(self, dfcoeff, savedir_coeff_heatmap=None, suffix=None):
        """
        Helper to plot dfcoeff, which holds the the coefficients of the regression for each chan, where the coefficients are
        the regression coefficients for each variable.
        """
        from pythonlib.tools.snstools import heatmap
        fig, _, _ = heatmap(dfcoeff, annotate_heatmap=False, diverge=True, labels_col=dfcoeff.columns, labels_row=self.Chans)         
        if savedir_coeff_heatmap is not None:
            if suffix is not None:
                savefig(fig, f"{savedir_coeff_heatmap}/regression_coeffs-{suffix}.pdf")
            else:
                savefig(fig, f"{savedir_coeff_heatmap}/regression_coeffs.pdf")

    def regress_neuron_task_variables_all_chans_data_splits(self, variables, variables_is_cat, 
                                                            var_effect_within_split, var_other_for_split):
        """
        For each chan, do multiple regression, where variables predicts firing rate.
        Here, "data_splits" means that can compute regression axes using different splits of dataset, each a level of vars_others, and
        then concatenate the results across all splits.
        - e.g, useful if want to compute axis for chunk_within_rank, conditioned on each level of chunk_shape.
        - also useful if you want to aggregate axes across different levels, to get a single better estimate of the axis.
        PARAMS:
        - variables, list of str, variables, to input input multipel regression.
        - variables_is_cat, list of bool, not used in regression, but used in later code.
        """    

        # tbin_dur = 0.2
        # tbin_slide = 0.1
        # npcs_keep_force = 50
        # normalization = "orthonormal"
        PLOT_COEFF_HEATMAP = False
        demean = False # Must be false, as we dont want function to modify PA
        get_axis_for_categorical_vars = True

        PAscal = self.norm_subtract_mean_each_chan()
        assert PAscal.X.shape[2]==1, "must be scalra"

        dflab = PAscal.Xlabels["trials"]
        # var_subspace = "chunk_within_rank"

        ### For each lev_other, extract the axis encoding <var_effect_within_split>
        list_axes = []
        # list_subspace_tuples = []
        levs_other = dflab[var_other_for_split].unique()
        for levo in levs_other:

            # First, get subset of data
            filtdict = {var_other_for_split:[levo]}
            pathis = PAscal.slice_by_labels_filtdict(filtdict)

            list_subspaces = []
            # list_subspaces = [(var_effect_within_split,)]
            _, _, _, dfcoeff, _, dfbases = pathis.dataextract_subspace_targeted_pca(
                            variables, variables_is_cat, list_subspaces, demean=demean, 
                            # normalization=normalization,
                            PLOT_COEFF_HEATMAP=PLOT_COEFF_HEATMAP, savedir_coeff_heatmap=None, PRINT=False,
                            get_axis_for_categorical_vars=get_axis_for_categorical_vars)

            list_axes.append(dfcoeff[var_effect_within_split])

        ### Store in a dataframe, the columns encoding <var_effect_within_split> within each levo (split)
        columns_each_split = [(var_effect_within_split, levo) for levo in levs_other]
        dfcoeff_splits = pd.DataFrame(np.stack(list_axes, axis=1), columns=columns_each_split)
        
        ### Rerun, using the entire data, not just the splits
        _, _, _, dfcoeff_all, _, dfbases = PAscal.dataextract_subspace_targeted_pca(
                            variables, variables_is_cat, [], demean=demean, 
                            # normalization=normalization, 
                            plot_orthonormalization=False, 
                            PLOT_COEFF_HEATMAP=False, savedir_coeff_heatmap=None, PRINT=False, get_axis_for_categorical_vars=get_axis_for_categorical_vars)

        ### Finally, merge the split and the all
        for col in dfcoeff_splits.columns:
            assert col not in dfcoeff_all.columns
        DFCOEFF = pd.concat([dfcoeff_all, dfcoeff_splits], axis=1)    

        # Also get the names of columns that are for the split levels:

        return DFCOEFF, columns_each_split

    def regress_neuron_task_variables_all_chans(self, variables, variables_is_cat, PLOT_COEFF_HEATMAP=False, PRINT=False,
                                                savedir_coeff_heatmap=None, get_axis_for_categorical_vars=False,
                                                savedir_pca_subspaces=None):
        """
        For each chan, do multiple regression, where variables predicts firing rate.
        """    

        assert len(variables) == len(variables_is_cat)

        res = []
        res_all = []
        for chan_idx in range(len(self.Chans)):
            dict_coeff, model, data, original_feature_mapping = self.regress_neuron_task_variables(chan_idx, variables, 
                                                                                                variables_is_cat, PRINT=PRINT)
            res.append(dict_coeff)
            res_all.append({
                "dict_coeff":dict_coeff,
                "model":model,
                "data":data,
                "original_feature_mapping":original_feature_mapping,
            })
        dfcoeff = pd.DataFrame(res)

        # Before get basis vectors, for categorical variables, get a single vector (first PC)
        if get_axis_for_categorical_vars:
            print(dfcoeff.columns.tolist())
            for var_subspace, var_is_cat in zip(variables, variables_is_cat):
                print(var_subspace, var_is_cat)
                if var_is_cat: 
                    from neuralmonkey.analyses.state_space_good import dimredgood_pca
                    # Get the levels for this categorical variable.
                    original_feature_mapping = res_all[0]["original_feature_mapping"]
                    list_var_inner = [k for k, v in original_feature_mapping.items() if v==var_subspace]

                    if len(list_var_inner)==0:
                        # This usually means this variable did not have enough variation to be part of regression model.
                        # print(list_var_inner)
                        # print("Var mappings:")
                        # for k, v in original_feature_mapping.items():
                        #     print(k, v)
                        # print("Failed to find this var: ", var_subspace)
                        # assert False
                        continue
                    
                    # Do PCA to get the first PC
                    data = dfcoeff.loc[:, list_var_inner].values
                    assert len(data)>0

                    Xpcakeep, _, _ = dimredgood_pca(data, method="sklearn")
                                                        #  plot_pca_explained_var_path="/tmp/test1.pdf", plot_loadings_path="/tmp/test2.pdf")
                    dfcoeff[var_subspace] = Xpcakeep[:, 0]

                    # Also, optionally, get not just first PC

        if PLOT_COEFF_HEATMAP:
            self.regress_neuron_task_variables_all_chans_plot_coeffs(dfcoeff, savedir_coeff_heatmap)
            # from pythonlib.tools.snstools import heatmap
            # fig, _, _ = heatmap(dfcoeff, annotate_heatmap=False, diverge=True, labels_col=dfcoeff.columns, labels_row=self.Chans)         
            # if savedir_coeff_heatmap is not None:
            #     savefig(fig, f"{savedir_coeff_heatmap}/regression_coeffs.pdf")

        # Get mapping from input variables to their levels (i.e,. their coefficent names)
        original_feature_mapping = res_all[0]["original_feature_mapping"]
        for x in res_all:
            assert original_feature_mapping == x["original_feature_mapping"], "failed sanity check! diff chanels have diff mapping, why?"

        ### Get multi-D subspace for each categorical variable. 
        # Get a subspace that is higher-D than just 1-D
        res = []
        for var_subspace, var_is_cat in zip(variables, variables_is_cat):
            if var_is_cat: 
                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                # Get the levels for this categorical variable.
                list_var_inner = [k for k, v in original_feature_mapping.items() if v==var_subspace]

                if len(list_var_inner)==0:
                    # This usually means this variable did not have enough variation to be part of regression model.
                    # print(list_var_inner)
                    # print("Var mappings:")
                    # for k, v in original_feature_mapping.items():
                    #     print(k, v)
                    # print("Failed to find this var: ", var_subspace)
                    # assert False
                    continue
                
                # Do PCA to get the first PC
                data = dfcoeff.loc[:, list_var_inner].values
                assert len(data)>0

                print(f"For var={var_subspace}, this many levels: {len(list_var_inner)}")
                if savedir_pca_subspaces is not None:
                    plot_pca_explained_var_path = f"{savedir_pca_subspaces}/pca-var_explained-{var_subspace}.pdf"
                    plot_loadings_path = f"{savedir_pca_subspaces}/pca-loadings-{var_subspace}.pdf"
                else:
                    plot_pca_explained_var_path, plot_loadings_path = None, None

                _, Xpca, _, explained_variance_ratio_, components_ = dimredgood_pca(data, method="sklearn", 
                                                plot_pca_explained_var_path=plot_pca_explained_var_path, 
                                                plot_loadings_path=plot_loadings_path, return_stats=True)

                # store everything
                res.append({
                    "var_subspace":var_subspace,
                    "var_is_cat":var_is_cat,
                    "list_var_inner":list_var_inner,
                    "Xpca":Xpca,
                    "explained_variance_ratio_":explained_variance_ratio_,
                    "components_":components_,
                })
        dfbases = pd.DataFrame(res)

        return dfcoeff, dfbases, res_all, original_feature_mapping
    
    def regress_neuron_task_variables(self, chan_idx, variables, variables_is_cat, PRINT=False):
        """
        For a single chan, do multiple regression, where variables predicts firing rate.
        Must be using scalar neural data
        """    
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        ### Average over time
        assert self.X.shape[2]==1, "you must pass in scalars -- e.g, could mean over time"
        # pa = self.slice_by_dim_values_wrapper("times", twind_scal)
        # pa = pa.agg_wrapper("times")

        ### Pull out chan
        frates = self.X[chan_idx, :, 0] # (ntrials, )

        ### Feeatures
        dflab = self.Xlabels["trials"]
        # TODO: Check for correlated variables.
        data = dflab.loc[:, variables].copy()
        data["frate"] = frates

        ### Construct function string
        # list_feature_names = []
        func = f"frate ~"
        for var, var_is_cat in zip(variables, variables_is_cat):
            if var_is_cat == False:
                func += f" {var} + "
                # list_feature_names.append(var)
        for var, var_is_cat in zip(variables, variables_is_cat):
            if var_is_cat == True:
                func += f" C({var}) + "
                # list_feature_names.append(var)
        # remove the + at the end
        func = func[:-3]

        ### Run regression
        model = smf.ols(func, data=data).fit()

        # Extract the coefficients
        feature_names = model.params.index.tolist()
        coef_array = model.params.values  # shape (1, nfeat)

        dict_coeff = {f:c for f, c in zip(feature_names, coef_array)}

        # Map from dummy variables back to original variables
        original_feature_mapping = {}
        for feat in feature_names:
            if 'C(' in feat:
                # e.g., feat = 'C(gender)[T.male]'
                base = feat.split('[')[0]  # 'C(gender)'
                base = base.replace('C(', '').replace(')', '')  # 'gender'
                original_feature_mapping[feat] = base
            else:
                original_feature_mapping[feat] = feat

        if PRINT:
            print(model.summary())
            print(feature_names)
            print(coef_array)   

        return dict_coeff, model, data, original_feature_mapping
        
def concatenate_popanals_flexible(list_pa, concat_dim="trials", how_deal_with_different_time_values="replace_with_dummy"):
    """ Concatenates popanals (along a given dim) which may have different time bases (but
    the same n time bins.
    If differnet, then replaces time with 0,1, 2... (index), otherwise uses actual time.
    PARAMS:
        - how_deal_with_different_time_values, str, if concatting along trials, and each pa has different
        timebase, then different methods for dealing with fact that the returned PA must have the same time labels
        across all datapts.
    RETURNS:
        - PA, new popanal
        - twind, (tmin, tmax) from new PA.Times.
    """
    assert len(list_pa)>0, "didnt get any data"

    if concat_dim=="trials":
        # if you are combining multiple times, then replace times iwth a
        # dummy variable
        times_identical = check_identical_times(list_pa)
        if not times_identical:
            if how_deal_with_different_time_values=="replace_with_dummy":
                # (0,1,2, ...)
                replace_times_with_dummy_variable = True
                all_pa_inherit_times_of_pa_at_this_index = None
                times_realign_so_first_index_is_this_time = None
            elif how_deal_with_different_time_values=="replace_with_first_pa":
                # Take the timestamps from first pa
                replace_times_with_dummy_variable = False
                all_pa_inherit_times_of_pa_at_this_index = 0
                times_realign_so_first_index_is_this_time = None
            elif how_deal_with_different_time_values=="replace_with_first_pa_realigned":
                # Take the timestamps from first pa, and then shift times so that
                # gtime of first time index is 0.
                replace_times_with_dummy_variable = False
                all_pa_inherit_times_of_pa_at_this_index = 0
                times_realign_so_first_index_is_this_time = 0.
            else:
                print(how_deal_with_different_time_values)
                assert False, "different time bases, not sure hwo to deal"
        else:
            replace_times_with_dummy_variable = False
            all_pa_inherit_times_of_pa_at_this_index = None
            times_realign_so_first_index_is_this_time = None

        PA = concatenate_popanals(list_pa, "trials",
                                    replace_times_with_dummy_variable=replace_times_with_dummy_variable,
                                    all_pa_inherit_times_of_pa_at_this_index=all_pa_inherit_times_of_pa_at_this_index,
                                    times_realign_so_first_index_is_this_time=times_realign_so_first_index_is_this_time)

    elif concat_dim=="times":
        # ALl will copy the trials df from the first pa.
        PA = concatenate_popanals(list_pa, "times",
                                     all_pa_inherit_trials_of_pa_at_this_index=0,
                                     replace_times_with_dummy_variable=False)

    elif concat_dim=="chans":
        # for pa in list_pa:
        #     print(pa.Chans) 
        PA = concatenate_popanals(list_pa, "chans",
                                  all_pa_inherit_trials_of_pa_at_this_index=0)
        # print("output")
        # print(PA.Chans)
        # assert False
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
     all_pa_inherit_trials_of_pa_at_this_index=None,
     times_realign_so_first_index_is_this_time=None):
    """ Concatenate multiple popanals. They must have same shape except
    for the one dim concatted along.
    PARAMS:
    - list_pa, list of PopAnal objects
    - dim, int, which dimensiion to concat along
    - values_for_concatted_dim, list of items which are labels for
    each value in the new concatted dimension. Must be apporopriate length. If None, thne
    concatenates the values in the input pa
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
    - times_realign_so_first_index_is_this_time, float, if not None, then forces that the first
    time index takes this value, by shifting times (subtraction). e.g,, if 0., then all times
    are shifted, maintaing same time period. Applies whether or not replace_times_with_dummy_variable is True
    RETURNS:
    - PopAnal object,
    --- or None, if inputed list_pa is empty.
    NOTE:
    - for the non-concatted dim, will use the values for the first pa. assumes
    this is same across pa.
    """
    import copy
    from pythonlib.tools.pandastools import concat

    if len(list_pa)==0:
        return None

    ### Initialize by making copies.
    list_pa = [pa.copy() for pa in list_pa]
    dim, dim_str = help_get_dimensions(dim)

    ### Fix problem where times are len 1 off from each other. 
    # (Sometimes times are len 1 off from each other. Here is quick fix.
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

    ### Deal with different time-bases across PAs
    if all_pa_inherit_times_of_pa_at_this_index is not None:
        assert replace_times_with_dummy_variable==False
        pa_base = list_pa[all_pa_inherit_times_of_pa_at_this_index]
        for pa in list_pa:
            assert len(pa.Times)==len(pa_base.Times)
            pa.Times = copy.copy(pa_base.Times)

    if replace_times_with_dummy_variable:
        assert all_pa_inherit_times_of_pa_at_this_index is None
        for pa in list_pa:
            pa.Times = np.arange(len(pa.Times))

    if times_realign_so_first_index_is_this_time is not None:
        for pa in list_pa:
            pa.Times = pa.Times - pa.Times[0] + times_realign_so_first_index_is_this_time
        
    # 2) Create new PA
    # Extract values to populate the other dimensions
    # - decide whether to enforce same values across all list_pa.
    check_this_dim = {}

    for d in ["times", "chans", "trials"]:
        check_this_dim[d] = (assert_otherdims_have_same_values) and (d in assert_otherdims_restrict_to_these)

    # - extract values
    if dim_str=="times":
        times = values_for_concatted_dim
        if times is None:
            # Make times = [(i, t) ...] where i is PA index, and t is the time iun PA.times.
            times =[]
            for i, patmp in enumerate(list_pa):
                # times.extend([(i, t) for t in patmp.Times])
                times.extend([f"{i}|{t:.4f}" for t in patmp.Times])
        chans = check_get_common_values_this_dim(list_pa,"chans", check_this_dim["chans"])
        trials = check_get_common_values_this_dim(list_pa, "trials", check_this_dim["trials"])
    elif dim_str=="chans":
        times = check_get_common_values_this_dim(list_pa, "times", check_this_dim["times"])
        if values_for_concatted_dim is None:
            _chans = []
            for _pa in list_pa:
                _chans.extend(_pa.Chans)
            chans = _chans
        else:
            chans = values_for_concatted_dim
        assert len(set(chans)) == len(chans), "have non-unique chans, this can run into error later"
        trials = check_get_common_values_this_dim(list_pa, "trials", check_this_dim["trials"])
    elif dim_str=="trials":
        times = check_get_common_values_this_dim(list_pa, "times", check_this_dim["times"])
        chans = check_get_common_values_this_dim(list_pa, "chans", check_this_dim["chans"])

        if values_for_concatted_dim is None:
            _trials = []
            for _pa in list_pa:
                _trials.extend(_pa.Trials)
            trials = _trials
        else:
            trials = values_for_concatted_dim   
            assert len(set(trials)) == len(trials), "have non-unique chans, this can run into error later"
    else:
        print(dim_str)
        assert False

    ### 1) Concat the data
    # Generate new popanal
    list_x = [pa.X for pa in list_pa]
    X = np.concatenate(list_x, axis=dim)
    PA = PopAnal(X, times=times, trials = trials,
        chans = chans)

    # Concatenate Xlabels dataframe
    # - concat the dimension chosen
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
        assert len(dftmp) == PA.X.shape[1], "prob because concateed along trials, so trails expanded.."
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
    from pythonlib.tools.checktools import check_objects_identical

    times_prev = None
    times_identical = True
    for pa in list_pa:
        if times_prev is not None:
            times_identical = check_objects_identical(pa.Times, times_prev)
            if not times_identical:
                break
            # if len(pa.Times) != len(times_prev):
            #     times_identical = False
            #     break
            # if not isin_array(pa.Times, [times_prev]):
            #     times_identical = False
            #     break
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

