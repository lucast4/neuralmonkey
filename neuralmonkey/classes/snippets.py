import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_snippets(sdir, fname="Snippets"):
    import pickle as pkl

    path = f"{sdir}/{fname}.pkl"
    with open(path, "rb") as f:
        SP = pkl.load(f)

    return SP


class Snippets(object):
    """
    Neural snippets, extraction of PopAnal objects in relation to different events, and 
    methods for reslicing, agging, plotting, etc.
    Written when doing primsinvar feature representations, but can apply generally 
    to most analyses.
    """

    def __init__(self, SN, which_level, list_events, 
        list_features_extraction, list_features_get_conjunction, 
        list_pre_dur, list_post_dur,
        strokes_only_keep_single=False,
        tasks_only_keep_these=None,
        prune_feature_levels_min_n_trials=10, 
        dataset_pruned_for_trial_analysis=None,
        trials_prune_just_those_including_events=True):
        """ Initialize a dataset
        PARAMS:
        - SN, Sessions object, holding neural data for a single session
        - which_level, str, what level is represented by each datapt {'trial', 'stroke'}
        - list_events, list of str, the events for each to extract PA for.
        - list_features_extraction, list of str, features which will extract for each trial or stroke
        (and assign to that datapt). NOTE: by default gets many features.. this is just
        extra. This is just for exgtraciton, not for plotting.
        - list_features_get_conjunction, list of str, features for which will get conjuction
        of other features. Thesea re also the features that wil be plot and anlyszes. 
        - dataset_pruned_for_trial_analysis, Dataset object, which,if not None, will
        be used to determine which trials (i.e. onluy tjhose in Dataset)            
        NOTE: see extraction_helper() for notes on params.
        """

        assert trials_prune_just_those_including_events==True, "this on by defualt. if turn off, then change line below in SN.get_trials_list"
        if dataset_pruned_for_trial_analysis is None:
            dataset_pruned_for_trial_analysis = SN.Datasetbeh

        self.DfScalar = None
        self.SN = SN

        # 1b) Which sites to use?
        sites = SN.sitegetter_all()
        if False:
            # Prune to top N sites, just for quick analysis
            N = 40
            sites_keep = SN._sitegetter_sort_sites_by(sites, "fr", N)
        elif False:
            # get the lowest fr sites
            sites_sorted = SN._sitegetter_sort_sites_by(sites, "fr")
            sites_keep = sites_sorted[:-N:-1]
        else:
            # Keep all sites
            sites_keep = sites
            print("\n == extarcating these sites: ", sites_keep)


        ### EXTRACT SNIPPETS
        if which_level=="stroke":
            # Sanity checks
            assert len(list_events)==0, "event is stroke. you made a mistake (old code, site anova?)"
            assert len(list_pre_dur)==1
            assert len(list_post_dur)==1

            pre_dur=list_pre_dur[0]
            post_dur = list_post_dur[0]

            # Each datapt matches a single stroke
            DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, 
                strokes_only_keep_single, tasks_only_keep_these, 
                prune_feature_levels_min_n_trials, 
                list_features_extraction)
            trials = None

            ListPA = self.extract_snippets_strokes(DS, sites_keep, pre_dur, post_dur,
                features_to_get_extra=list_features_extraction)

            # Fill in dummy variables
            list_events = ["stroke"]
            list_events_uniqnames = ["00_stroke"]

        elif which_level=="trial":
            # Each datapt is a single trial
            # no need to extract antyhing, use sn.Datasetbeh
            # only those trials that exist in SN.Datasetbeh
            trials = SN.get_trials_list(True, True, only_if_in_dataset=True, 
                dataset_input=dataset_pruned_for_trial_analysis,
                events_that_must_include = list_events)
            print("\n == extracting these trials: ", trials)
            DS = None

            assert len(list_events) == len(list_pre_dur)
            assert len(list_events) == len(list_post_dur)

            ListPA, list_events_uniqnames = self.extract_snippets_trials(trials, sites_keep, list_events, list_pre_dur, list_post_dur,
                list_features_extraction)
        else:
            assert False

        # Map var conjunctions.
        for pa in ListPA:
            map_var_to_othervars = pa.labels_features_input_conjunction_other_vars(dim="trials", 
                list_var = list_features_get_conjunction)

        ### SAVE VARIABLES
        self.ListPA = ListPA
        self.Sites = sites_keep
        self.DS = DS
        self.Trials = trials
        self.Params = {
            "which_level":which_level,
            "list_events":list_events, 
            "list_events_uniqnames":list_events_uniqnames,
            "list_features_extraction":list_features_extraction,
            "list_features_get_conjunction":list_features_get_conjunction,
            "list_pre_dur":list_pre_dur,
            "list_post_dur":list_post_dur,
            "map_var_to_othervars":map_var_to_othervars,
            "strokes_only_keep_single":strokes_only_keep_single,
            "tasks_only_keep_these":tasks_only_keep_these,
            "prune_feature_levels_min_n_trials":prune_feature_levels_min_n_trials,
        }

        # Genreate scalars
        if False:
            # Old version
            print("\n == listpa_convert_to_scalars")        
            self.listpa_convert_to_scalars()
            print("\n == pascal_convert_to_dataframe")        
            self.pascal_convert_to_dataframe(fr_which_version="sqrt")
        else:
            self.listpa_convert_to_scalar_v2(fr_which_version="sqrt")
            self.DfScalar = self.DfScalar # they are the same

        self.pascal_remove_outliers()

        # Get useful variables
        print("\n == _preprocess_map_features_to_levels")        
        self._preprocess_map_features_to_levels()
        self._preprocess_map_features_to_levels_input("character")

        print(f"** Generated Snippets, (ver {which_level}). Final length of SP.DfScalar: {len(self.DfScalar)}")


    def extract_snippets_strokes(self, DS, sites_keep, pre_dur, post_dur,
            features_to_get_extra=None):
        """ Extract popanal, one data pt per stroke
        """

        # 2) Extract snippets
        ListPA = self.SN.popanal_generate_alldata_bystroke(DS, sites_keep, align_to_stroke=True, 
                                                      align_to_alternative=[], 
                                                      pre_dur=pre_dur, post_dur=post_dur,
                                                      use_combined_region=False,
                                                      features_to_get_extra=features_to_get_extra)
        assert len(ListPA)==1
        return ListPA

    def extract_snippets_trials(self, trials, sites_keep, list_events, list_pre_dur, list_post_dur,
            list_features_extraction):
        """
        Each snippet is a single trial x event. 
        This only keeps trials that have each event in the list of events

        """

        # Must do this, or else will bug below, since wont keep self.Xlabels["trial"]
        for col in ("trialcode", "epoch", "character", "supervision_stage_concise"):
            if col not in list_features_extraction:
                list_features_extraction.append(col)

        # assert len(list_features_extraction)>0, "or else downstream will fail by not extracting "
        # 2) Extract snippets
        ListPA = []
        fig, ax = plt.subplots()
        list_events_uniqnames = []
        map_var_to_othervars_list = []
        for i, (event, pre_dur, post_dur) in enumerate(zip(list_events, list_pre_dur, list_post_dur)):
            
            # 1) Extract single pa
            print("\n == generating popanal for: ", event)
            listpa = self.SN.popanal_generate_alldata(trials, sites_keep,
                events = [event],
                pre_dur=pre_dur, post_dur=post_dur, 
                columns_to_input = list_features_extraction,
                use_combined_region = False)
            assert len(listpa)==1
            pa = listpa[0]

            # # 2) Get conjuctions of features
            # print("\n == labels_features_input_conjunction_other_vars: ", event)
            # map_var_to_othervars = pa.labels_features_input_conjunction_other_vars(dim="trials", 
            #     list_var = list_features_get_conjunction)

            ListPA.append(pa)
            
            # plot
            ax.plot([pre_dur, post_dur], [i, i], "o-", label=event)
            
            # give event a unique name
            # Version 2, with indices
    #         event_unique_name = f"{i}_{event[:3]}_{event[-1]}"
            if i<10:
                idx_str = f"0{i}"
            else:
                idx_str = f"{i}"
            event_unique_name = f"{idx_str}_{event}"
            list_events_uniqnames.append(event_unique_name)

        ax.set_title('Time windows extracted')
        ax.legend()
        ax.set_xlabel('time, rel event (sec)')
        ax.axvline(0)
        print("* List events:", list_events_uniqnames)      

        return ListPA, list_events_uniqnames

    def _preprocess_map_features_to_levels(self):
        """ Generate a single mapper from features to its levels, 
        that can apply across all data in self. 
        will be sorted.
        """
        list_var = self.Params["list_features_get_conjunction"]
        data = self.DfScalar

        MapVarToLevels = {} # to have consistent levels.
        for var in list_var:
            levels = sorted(data[var].unique().tolist())
            MapVarToLevels[var] = levels

        self.Params["map_var_to_levels"] = MapVarToLevels

    def _preprocess_map_features_to_levels_input(self, feature):
        """ Get the levels for this feature and save it into self.Params["map_var_to_levels"]
        PARAMS
        - feature, string, will get all its levels.
        RETURNS:
        - self.Params["map_var_to_levels"] updated
        """

        if feature in self.DfScalar.columns:
            if feature not in self.Params["map_var_to_levels"]:
                levels = sorted(self.DfScalar[feature].unique().tolist())
                self.Params["map_var_to_levels"][feature] = levels


    ############################################# WORKING WITH POPANALS
    def popanal_extract_specific_slice(self, event_uniq, chan=None, var_level=None):
        """ Extract a specific slice of popanal
        PARAMS:
        - event_uniq, unique name, usually number-prefixed.
        - chan, channel (value) to keep
        - var_level, list-like, 2 values, (var, level), to keep just this level for
        this var in pa.Xlabels["trials"]
        """

        # Get for this event
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        i_event = list_events_uniqnames.index(event_uniq)
        pa = self.ListPA[i_event]

        # Get this chan
        if chan is not None:
            pa = pa.slice_by_dim_values_wrapper("chans", [chan])

        # Get for this level of var
        if var_level is not None:
            var = var_level[0] # str
            lev = var_level[1] # value
            pa = pa.slice_by_label("trials", var, lev)

        return pa

    ############################################ SCALARS
    def listpa_convert_to_smfr(self):
        """ Get a single dataframe holding all smoothed fr traces.
        RETURNS:
        - stores in self.DfScalar
        """
        from neuralmonkey.classes.population import concatenate_popanals

        list_events_uniqnames = self.Params["list_events_uniqnames"]
        assert len(list_events_uniqnames)==len(self.ListPA)

        # == 1) Concatenate across pa (events) (and scalarize)
        list_df = []
        for pa, event in zip(self.ListPA, list_events_uniqnames):
            # extract df
            dfthis = pa.convert_to_dataframe_long()
            dfthis["event_aligned"] = event
            list_df.append(dfthis)

        self.DfScalar = pd.concat(list_df).reset_index(drop=True)

    def listpa_convert_to_scalar_v2(self, fr_which_version="raw"):
        """ For each trial, get a single scalar value by averaging across time
        NEw version (2/23/23) first extract all fr, then does mean over time this 
        good because keeps scalar and sm fr in same dataframe. 
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        if self.DfScalar is None:
            self.listpa_convert_to_smfr() 

        # take mean over time
        def F(x):
            # assert len(x["fr_sm"].shape)==2 
            # return np.mean(x["fr_sm"], axis=1)
            return np.mean(x["fr_sm"])

        self.DfScalar = applyFunctionToAllRows(self.DfScalar, F, "fr_scalar_raw")

        # tgransform the fr if desired
        if fr_which_version=="raw":
            self.DfScalar["fr_scalar"] = self.DfScalar["fr_scalar_raw"] 
        elif fr_which_version=="sqrt":
            self.DfScalar["fr_scalar"] = self.DfScalar["fr_scalar_raw"]**0.5
        else:
            print(fr_which_version)
            assert False

        self.DfScalar["fr_sm_sqrt"] = self.DfScalar["fr_sm"]**0.5

    def listpa_convert_to_scalars(self):
        """ For each trial, get a single scalar value by averaging across time
        """
        ### Collect scalars across all pa
        # Features across which data will vary: (all stroke features), (events), (chans)

        from neuralmonkey.classes.population import concatenate_popanals

        list_events_uniqnames = self.Params["list_events_uniqnames"]

        # == 1) Concatenate across pa (events) (and scalarize)
        list_pa_scal = []
        for pa in self.ListPA:
            # 1. given a pa, compute scalar for each row
            pascal = pa.agg_wrapper("times", "mean", rename_values_agged_dim=False)
            list_pa_scal.append(pascal)

        # concatenate
        PA_scal_all = concatenate_popanals(list_pa_scal, dim="trials", 
                                          map_idxpa_to_value=list_events_uniqnames, 
                                          map_idxpa_to_value_colname="event_aligned")

        # == Flatten --> split each chan to smaller pa, then concatenate 
        PA_scal_all = PA_scal_all.reshape_by_splitting()
        # list_pa_scal = []
        # list_labels = []
        # for i in range(len(PA_scal_all.Chans)):
        #     pa_scal = PA_scal_all.slice_by_dim_indices_wrapper("chans", [i])
        #     chan = PA_scal_all.Chans[i]
        #     list_pa_scal.append(pa_scal)
        #     list_labels.append(chan)
        # PA_scal_all= concatenate_popanals(list_pa_scal, dim="trials", 
        #                                 map_idxpa_to_value=list_labels, 
        #                                 map_idxpa_to_value_colname="chan",
        #                                 assert_otherdims_have_same_values=False)

        # Print summary of the final scalar values
        print("Final data shape:", PA_scal_all.X.shape)
        self.PAscalar = PA_scal_all

    def pascal_convert_to_dataframe(self, fr_which_version = "raw"):
        """
        Convert self.PAscalar to dataframe, which can then be used for analyses, etc, easily
        PARAMS:
        - fr_which_version, str, which version of firing rate to use, {'raw', 'sqrt'}, will
        assign this to df["fr_scalar"]. Note, raw will always be in df["fr_scalar_raw"]
        RETURNS:
        - modifies self.DfScalar and returns it
        """

        print("Running SP.pascal_convert_to_dataframe")

        # Convert from PA to a single dataframe that includes scalar FR
        df = self.PAscalar.Xlabels["trials"]
        df["fr_scalar_raw"] = self.PAscalar.X.squeeze()

        if fr_which_version=="raw":
            df["fr_scalar"] = df["fr_scalar_raw"] 
        elif fr_which_version=="sqrt":
            df["fr_scalar"] = df["fr_scalar_raw"]**0.5
        else:
            print(fr_which_version)
            assert False

        self.DfScalar = df
        
        print("len self.DfScalar:", len(df))
        print("self.PAscalar.X.shape : ", self.PAscalar.X.shape)
        return self.DfScalar

    def pascal_remove_outliers(self):
        """ Remove outliers based on fr_scalar, only for high fr outliers,
        defualt is 3.5x std + mean.
        RETURNS:
        - modifies self.DfScalar, removing outliers.
        also saves the old in self.DfScalar_BeforeRemoveOutlier
        NOTE: will do diff thing every time run.
        """
        from pythonlib.tools.pandastools import aggregThenReassignToNewColumn, applyFunctionToAllRows

        if hasattr(self, "DfScalar_BeforeRemoveOutlier"):
            assert False, "Looks like already ran.. comment this out if you want to "
        
        df = self.DfScalar

        # 1) extract upper limit
        # def F(x):
        #     mu = np.mean(x["fr_scalar"])
        #     sd = np.std(x["fr_scalar"])
        #     lower = mu - 3.5*sd
        #     return lower
        # df = aggregThenReassignToNewColumn(df, F, "chan", "outlier_lims_lower")

        def F(x):
            mu = np.mean(x["fr_scalar"])
            sd = np.std(x["fr_scalar"])
            upper = mu + 3.5*sd
            return upper
        df = aggregThenReassignToNewColumn(df, F, ["chan"], "outlier_lims_upper")

        # for each row, is it in bounds?
        def F(x):
            return (x["fr_scalar"]>x["outlier_lims_upper"])
        
        # 2) each row. outlier?
        df = applyFunctionToAllRows(df, F, "outlier_remove")

        # 3) remove the outliers
        self.DfScalar_BeforeRemoveOutlier = df.copy()
        df = df[df["outlier_remove"]==False].reset_index(drop=True)
        self.DfScalar = df
        print("Starting len: ", len(self.DfScalar_BeforeRemoveOutlier))
        print("Removed outliers! new len: ", len(df))


    ############# working with frmat
    # sort sites by time of max fr
    def frmat_sort_by_time_of_max_fr(self, frmat, sites):
        """ Sorts sites (axis 0) so that top row
        has the earliset time of max fr
        RETURNS:
        - frmat_sorted, sites_sorted (COPIES)
        """
        max_fr_times = np.argmax(frmat, axis=1)
        inds_sort = np.argsort(max_fr_times)
        
        frmat_sorted = np.copy(frmat)
        sites_sorted = [s for s in sites]
        
        frmat_sorted = frmat_sorted[inds_sort]
        sites_sorted = [sites_sorted[i] for i in inds_sort]
        
        return frmat_sorted, sites_sorted


    ################ DATA EXTRACTIONS
    def dataextract_as_df_conjunction_vars(self, var, vars_others, site=None,
        n_min = 8):
        """ Helper to extract dataframe (i) appending a new column
        with ocnjucntions of desired vars, and (ii) keeping only 
        levels of this vars (vars_others) that has at least n trials for 
        each level of a var of interest (var).
        Useful becuase the output is ensured to have all levels of var, and (wiht 
        some mods) could be used to help balance the dataset.
        PARAMS:
        - var, str, the variabiel you wisht ot use this dataset to compute 
        moudlation for.
        - vars_others, list of str, variables that conjucntion will define a 
        new goruping var.
        - site, either None (all data) or int.
        - n_min, min n trials desired for each level of var. will only keep
        conucntions of (vars_others) which have at least this many for each evel of
        var.
        EG:
        - you wish to ask about shape moudlation for each combaiton of location and 
        size. then var = shape and vars_others = [location, size]
        RETURNS:
        - dataframe, with new column "vars_others"
        - dict, lev:df
        """
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        import numpy as np

        # 1) extract for this site.
        if site is None:
            dfthis = self.DfScalar
        else:
            dfthis = self.DfScalar[(self.DfScalar["chan"] == site)]

        # 2) extract_with_levels_of_conjunction_vars
        dfthis, dict_lev_df = extract_with_levels_of_conjunction_vars(dfthis, var, vars_others, n_min)

        return dfthis, dict_lev_df


    def dataextract_as_df(self, grouping_variables, grouped_var_col_name):
        from pythonlib.tools.pandastools import append_col_with_grp_index
        dfthis = append_col_with_grp_index(self.DfScalar, grouping_variables, 
            grouped_var_col_name, strings_compact=True)

        columns_keep = ["chan", "trialcode", "fr_sm"]
        return dfthis.loc[:, columns_keep + [grouped_var_col_name]]

    def dataextract_as_frmat(self, chan, event=None, var=None, var_level=None, 
        fr_ver="fr_sm", return_as_zscore=False):
        """ 
        Extract frmat from self.DfScalar, stacking all instances of this event, and
        (optionally) only this level for this var.
        PARAMS
        - chan, int
        - event, unique event (00_..) into event_aligned
        - var, var_level, either both None (ignore var), or string and value.
        RETURNS:
        - frmat, (ntrials, ntime)
        """
        
        assert (var_level==None) == (var==None)
        
        if event is None and var is None:
            dfthis = self.DfScalar[(self.DfScalar["chan"]==chan)]
        elif event is None:
            dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar[var]==var_level)]   
        elif var is None:
            dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar["event_aligned"]==event)]    
        else:
            dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar["event_aligned"]==event) & (self.DfScalar[var]==var_level)]   
            
        frmat = np.concatenate(dfthis[fr_ver].tolist(), axis=0)    
            
        if return_as_zscore:
            def _frmat_convert_zscore(frmat):
                """ convert to a single zscore trace, using the grand mean and std.
                Returns same shape as input
                """
                m = np.mean(frmat[:], keepdims=True)
                s = np.std(frmat[:], keepdims=True)

                return (frmat - m)/s
            frmat = _frmat_convert_zscore(frmat)

        return frmat 

    def dataextract_as_popanal(self, chan=None, events_uniq=None):
        """ Return a single popanal """
        assert False, "in progress"

    def _dataextract_as_metrics_scalar(self, dfthis):
        """ Pass in dfthis directly
        """
        from neuralmonkey.metrics.scalar import MetricsScalar
        
        # Input to Metrics
        # (use this, instead of auto, to ensure common values across all chans)
        list_var = self.Params["list_features_get_conjunction"]
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        map_var_to_othervars = self.Params["map_var_to_othervars"]
        map_var_to_levels = self.Params["map_var_to_levels"]
        Mscal = MetricsScalar(dfthis, list_var, map_var_to_othervars, 
            map_var_to_levels, 
            list_events_uniqnames)
        return Mscal

    # def dataextract_as_sm_fr_mat(self, chan)

    def dataextract_as_metrics_scalar(self, chan):
        """ Return scalar data for this chan, as a MetricsScalar object, which 
        has methods for computing modulation, etc
        """
        from neuralmonkey.metrics.scalar import MetricsScalar

        dfthis = self.DfScalar[self.DfScalar["chan"]==chan]
        return self._dataextract_as_metrics_scalar(dfthis)

        # # Input to Metrics
        # # (use this, instead of auto, to ensure common values across all chans)
        # list_var = self.Params["list_features_get_conjunction"]
        # list_events_uniqnames = self.Params["list_events_uniqnames"]
        # map_var_to_othervars = self.Params["map_var_to_othervars"]
        # map_var_to_levels = self.Params["map_var_to_levels"]
        # Mscal = MetricsScalar(dfthis, list_var, map_var_to_othervars, 
        #     map_var_to_levels, 
        #     list_events_uniqnames)
        # return Mscal

    # def dataextract_as_sm_fr_mat(self, chan)


    ################ MODULATION BY VARIABLES
    def modulationgood_compute_wrapper(self, var, vars_conjuction=None, list_site=None,
            n_min =8, score_ver="r2smfr_minshuff"):
        """ Good, flexible helper to compute modulation of all kinds and all ways of slicing 
        the dataset.
        PARAMS;
        - n_min, min n trials required for each level of var. if faisl, then skips this datset 
        entirely (i.e., the level of vars_conjuction, or this site)
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        import numpy as np

        # var = "chunk_rank"

        # Want to use entier data for this site? or do separately for each level of a given
        # conjunction variable.
        if vars_conjuction is None:
            # then place a dummy variable so that entire thing is one level
            vars_conjuction = ["dummy_var"]
            assert "dummy_var" not in self.DfScalar
            self.DfScalar["dummy_var"] = "IGNORE"
            # vars_conjuction = ['gridloc', 'chunk_within_rank'] # list of str, vars to take conjunction over

        if list_site is None:
            list_site = self.Sites

        list_events = self.Params["list_events_uniqnames"]
        # n_min =8
        # score_ver = "r2smfr_minshuff"
        # map_score_to_index_in_res = {
        #     "r2smfr_minshuff":3
        # }
        # INDEX_IN_RES = map_score_to_index_in_res[score_ver]

        # Collect data for each site
        OUT = []
        for site in list_site:
                    
            if site%20==0:
                print(site)
            region = self.SN.sitegetter_map_site_to_region(site)
            # Clean up dataset
            dfthis, _ = self.dataextract_as_df_conjunction_vars(var, vars_conjuction, site, n_min)
                
            if len(dfthis)==0:
                # then no level of vars_conjuction had enough data across all levels of var.
                continue

            # for each level of vars_conj, compute modulation
            levels_others = dfthis["vars_others"].unique().tolist()

            for lev in levels_others:
                
                # get data
                dfthisthis = dfthis[dfthis["vars_others"]==lev]
                
                # compute modulation
                MS = self._dataextract_as_metrics_scalar(dfthisthis)
                eventscores = MS.modulationgood_wrapper_(var, version=score_ver)
                
                for ev in list_events:
                    score = eventscores[ev]

                    # save
                    OUT.append({
                        "site":site,
                        "var":var,
                        "var_others":vars_conjuction,
                        "lev_in_var_others":lev,
                        "event":ev,
                        "score_ver":score_ver,
                        "score":score,
                        "region":region
                    })

        dfout = pd.DataFrame(OUT)

        if "dummy_var" in self.DfScalar:
            del self.DfScalar["dummy_var"]
        return dfout

    def modulation_compute_each_chan(self, DEBUG=False, 
        bregion_add_num_prefix=True, 
        bregion_combine=False, things_to_compute=("modulation", "fr")):
        """ Compute modulation by variables for each channel
        RETURNS:
        - RES_ALL_CHANS, list of dicts
        """

        RES_ALL_CHANS = []
        # list_chans = self.DfScalar["chan"].unique().tolist()
        list_chans = self.Sites

        if DEBUG:
            # faster...
            n = 10 # keep 10 chans
            inter = int(len(list_chans)/n)
            if inter==0:
                inter=1
            list_chans = list_chans[::inter]

        if bregion_add_num_prefix:
            # generate map from bregion to its number
            regions_in_order = self.SN.sitegetter_get_brainregion_list(bregion_combine)
            map_bregion_to_idx = {}
            for i, reg in enumerate(regions_in_order):
                map_bregion_to_idx[reg] = i

        for chan in list_chans:
            print("CHANNEL:", chan)
            info = self.SN.sitegetter_thissite_info(chan)

            Mscal = self.dataextract_as_metrics_scalar(chan)
            # dfthis = self.DfScalar[self.DfScalar["chan"]==chan]
            
            # # Input to Metrics
            # # (use this, instead of auto, to ensure common values across all chans)
            # list_var = self.Params["list_features_get_conjunction"]
            # list_events_uniqnames = self.Params["list_events_uniqnames"]
            # map_var_to_othervars = self.Params["map_var_to_othervars"]
            # map_var_to_levels = self.Params["map_var_to_levels"]
            # Mscal = MetricsScalar(dfthis, list_var, map_var_to_othervars, 
            #     map_var_to_levels, 
            #     list_events_uniqnames)
            
            # Compute, modulation across vars
            if "modulation" in things_to_compute:
                print("Mscal.modulation_calc_summary...")
                RES = Mscal.modulation_calc_summary()
            else: 
                RES = {}

            # Compute, fr across levels, for each var
            if "fr" in things_to_compute:
                print("Mscal.calc_fr_across_levels...")
                RES_FR = Mscal.calc_fr_across_levels()
            else:
                RES_FR = {}

            if DEBUG:
                print("======== RES")
                for k, v in RES.items():
                    print('---', k)
                    print(v)
                print("======== RES_FR")
                for k, v in RES_FR.items():
                    print('---', k)
                    print(v)
            
            # Merge them
            for k, v in RES_FR.items():
                assert k not in RES.keys()
                RES[k] = v

            # SAVE IT
            bregion = info["region"]
            if bregion_add_num_prefix:
                # for ease of sortinga nd plotting
                idx = map_bregion_to_idx[bregion]
                if idx<10:
                    idx_str = f"0{idx}"
                else:
                    idx_str = f"{idx}"
                bregion = f"{idx_str}_{bregion}"

            RES_ALL_CHANS.append({
                "chan":chan,
                "bregion":bregion,
                "RES":RES,
                "RES_FR": RES_FR
                })

        return RES_ALL_CHANS

    def modulation_compute_higher_stats(self, RES_ALL_CHANS):
        """ Compute higher (derived) stats
        PARAMS:
        - RES_ALL_CHANS, output from self.modulation_compute_each_chan
        """

        from scipy.stats import linregress as lr
        import pandas as pd
        import seaborn as sns

        MODULATION_FIELDS = ["modulation_across_events", 
                    "modulation_across_events_subgroups", 
                    "inconsistency_across_events",
                    "modulation_across_events_usingsmfr",
                    "modulation_across_events_usingsmfr_zscored"]

        list_var = self.Params["list_features_get_conjunction"]
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        # Get the list of methods for computing average modulation(across var)
        out = []
        for RES_ALL in RES_ALL_CHANS:
            list_meth = RES_ALL["RES"]["avgmodulation_across_methods_labels"]
            if list_meth not in out:
                out.append(list_meth)
        assert len(out)==1
        list_methods = out[0]

        # convert to dataframe to make plotting easier
        dat_across_var_events = []
        dat_across_var_methods = []
        dat_across_events = []
        dat_summary_mod = []
        dat_fr_across_events = []

        for RES_ALL in RES_ALL_CHANS:

            chan = RES_ALL["chan"]
            if "load_suffix" in RES_ALL:
                # Then this was loaded, note down which dataset it is
                load_suffix = RES_ALL["load_suffix"]
            else:
                load_suffix = "IGNORE"
            bregion = RES_ALL["bregion"]
            RES = RES_ALL["RES"]
            RES_FR = RES_ALL["RES_FR"]

            for var in list_var:   

                ########################################
                for val_kind in MODULATION_FIELDS:

                    y = RES[val_kind][var]
                    for ev, yscal in zip(list_events_uniqnames, y):
                        dat_across_var_events.append({
                            "event":ev,
                            "val":yscal,
                            "val_kind":val_kind,
                            "var":var,
                            "chan":chan,
                            "load_suffix":load_suffix,
                            "bregion":bregion
                        })
                                        
                ########################################
                # 4) avgmodulation_across_methods
                y = RES["avgmodulation_across_methods"][var]
                for method, yscal in zip(list_methods, y):
                    dat_across_var_methods.append({
                        "method":method,
                        "val":yscal,
                        "val_kind":"avgmodulation_across_methods",
                        "var":var,
                        "chan":chan,
                        "load_suffix":load_suffix,
                        "bregion":bregion
                    })
                
            # 4) avgmodulation_across_methods
            y = RES["avgfr_across_events"]
            for ev, yscal in zip(list_events_uniqnames, y):
                dat_across_events.append({
                    "event":ev,
                    "val":yscal,
                    "val_kind":"avgfr_across_events",
                    "chan":chan,
                    "load_suffix":load_suffix,
                    "bregion":bregion
                })

            ######################## MEAN FR ACROSS LEVELS
            # mod across events
            if len(RES_FR)>0:
                for ev_var, values in RES_FR["splitevents_alldata"].items():
                    ev = ev_var[0]
                    var = ev_var[1]
                    list_levels = self.Params["map_var_to_levels"][var]
                    assert len(values)==len(list_levels)    

                    for lev, yscal in zip(list_levels, values):
                        dat_fr_across_events.append({
                            "event":ev,
                            "val":yscal,
                            "val_kind":"raw",
                            "var_level":lev,
                            "var":var,
                            "chan":chan,
                            "load_suffix":load_suffix,
                            "bregion":bregion
                        })

                    # subtract mean over values
                    # (allows for easier comparison across neurons)
                    values_delt = np.array(values) - np.mean(values)
                    for lev, yscal in zip(list_levels, values_delt):
                        dat_fr_across_events.append({
                            "event":ev,
                            "val":yscal,
                            "val_kind":"minus_mean",
                            "var_level":lev,
                            "var":var,
                            "chan":chan,
                            "load_suffix":load_suffix,
                            "bregion":bregion
                        })



            
            ######################## DERIVED METRICS
            # A single "tuple" summarizing this neuron's mod
            modver = "modulation_across_events_subgroups"
        #     mod_tuple = ()
            idx_first_event = self._modulation_find_first_postsamp_event()
            for var in list_var:

                # 1) The mean modulation across events
                mod_mean = np.mean(RES[modver][var])
                dat_summary_mod.append({
                    "chan":chan,
                    "load_suffix":load_suffix,
                    "bregion":bregion,
                    "val":mod_mean,
                    "val_kind":f"{var}_mean",
                })
                
                # 2) Slope of modulation across events
                if False:
                    # works for when there are only 3 events...
                    mod_delt = RES[modver][var][2] - RES[modver][var][0]
                    mod_delt_norm = mod_delt/mod_mean
                else:
                    y = RES[modver][var][idx_first_event:] # start from first event.
                    x = np.arange(len(y))
                    mod_delt = lr(x, y).slope
                dat_summary_mod.append({
                    "chan":chan,
                    "load_suffix":load_suffix,
                    "bregion":bregion,
                    "val":mod_delt,
                    "val_kind":f"{var}_delt",
                })

        # Convert to dataframe
        dfdat_var_events = pd.DataFrame(dat_across_var_events)            
        dfdat_var_methods = pd.DataFrame(dat_across_var_methods)            
        dfdat_events = pd.DataFrame(dat_across_events)            
        dfdat_summary_mod = pd.DataFrame(dat_summary_mod)
        dfdat_fr_events = pd.DataFrame(dat_fr_across_events)

        OUT = {
            "dfdat_var_events":dfdat_var_events,
            "dfdat_var_methods":dfdat_var_methods,
            "dfdat_events":dfdat_events,
            "dfdat_summary_mod":dfdat_summary_mod,
            "dfdat_fr_events":dfdat_fr_events,
        }            

        return OUT

    def _modulation_find_first_postsamp_event(self, return_zero_if_fail=True):
        """ For computing slope of change in modulation over events, find 
        first event (chronolicaly) that is after presentaiton of the samp.
        ignore preceding events for computing slope
        RETURNS:
        - idx_event_firstpostsamp, int, index into list_events
        """

        list_events = self.Params["list_events"]
        list_pre_dur = self.Params["list_pre_dur"]
        list_post_dur = self.Params["list_post_dur"]

        # find the first event which is post-samp
        for i, (ev, pre, post) in enumerate(zip(list_events, list_pre_dur, list_post_dur)):
            if ev=="samp" and pre>0. and post>0.:
                # this is the first post-samp event
                print("Found first post-samp event: ", i, ev, "| times:", pre, post)
                idx_event_firstpostsamp = i
                return idx_event_firstpostsamp

        if return_zero_if_fail:
            return 0
        else:
            print(list_events)
            print(list_pre_dur)
            print(list_post_dur)
            assert False, "did not find..."
 
    def modulation_plot_heatmaps(self, OUT, savedir="/tmp",
            val_kind = "modulation_across_events_subgroups"):
        """ Plot heatmaps, bregion vs. event, and also
        overlay onto schematic of brain, across time(events)
        """
        from pythonlib.tools.pandastools import convert_to_2d_dataframe, aggregGeneral
        df = OUT["dfdat_var_events"]
        # val_kind = "modulation_across_events_subgroups"
        list_var = df["var"].unique().tolist()

        # 1) Plot modulation for each var
        DictDf = {}
        DictDf_rgba_values = {}
        norm_method = None
        # norm_method = "row_sub_firstcol"
        for var in list_var:
            dfthis = df[(df["val_kind"]==val_kind) & (df["var"]==var)]
            dfthis_agg = aggregGeneral(dfthis, group=["bregion", "event"], values=["val"])
        #     ZLIMS = [dfthis_agg["val"].min(), dfthis_agg["val"].max()]
            ZLIMS = [None, None]

            for annotate_heatmap in [True, False]:
                dfthis_agg_2d, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis_agg, 
                                                             "bregion", "event", True, agg_method="mean", 
                                                             val_name="val", 
                                                             norm_method=norm_method,
                                                             annotate_heatmap=annotate_heatmap,
                                                            zlims = ZLIMS
                                                            )
                ax.set_title(var)
                DictDf[var] = dfthis_agg_2d
                DictDf_rgba_values[var] = rgba_values

                # save fig
                fig.savefig(f"{savedir}/1-{val_kind}-{var}-annot_{annotate_heatmap}.pdf")

        # 2) Plot heatmap of difference between two variables
        from pythonlib.tools.snstools import heatmap
        if len(list_var)>1:
            var0 = list_var[0]
            var1 = list_var[1]
            dfthis_2d = DictDf[var1] - DictDf[var0]
        #     ZLIM = [None, None]
            zmax = dfthis_2d.abs().max().max()
            ZLIMS = [-zmax, zmax]
            for annotate_heatmap in [True, False]:
                fig, ax, rgba_values = heatmap(dfthis_2d, annotate_heatmap=annotate_heatmap, 
                    zlims=ZLIMS, diverge=True)
                ax.set_title(f"{var1}-min-{var0}")
                # Save
                DictDf[f"{var1}-min-{var0}"] = dfthis_2d
                DictDf_rgba_values[f"{var1}-min-{var0}"] = rgba_values     

                # save fig
                fig.savefig(f"{savedir}/2-{val_kind}-{var1}-min-{var0}-annot_{annotate_heatmap}.pdf")

                    
        # 3) Average fr
        # norm_method = "row_sub_firstcol"
        df = OUT["dfdat_events"]
        dfthis = df[(df["val_kind"]=="avgfr_across_events")]
        dfthis_agg = aggregGeneral(dfthis, group=["bregion", "event"], values=["val"])

        # for norm_method, diverge in zip(["row_sub_firstcol", None], [True, False]):
        for norm_method, diverge in zip([None], [False]):
            for annotate_heatmap in [True, False]:
                dfthis_agg_2d, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis_agg, 
                                                             "bregion", "event", True, agg_method="mean", 
                                                             val_name="val", 
                                                             norm_method=norm_method,
                                                             annotate_heatmap=annotate_heatmap,
                                                             diverge=diverge
                                                            )

                DictDf[f"avgfr_across_events-norm_{norm_method}"] = dfthis_agg_2d
                DictDf_rgba_values[f"avgfr_across_events-norm_{norm_method}"] = rgba_values

                # save fig
                fig.savefig(f"{savedir}/3-avgfr_across_events-annot_{annotate_heatmap}.pdf")

        return DictDf, DictDf_rgba_values


    def modulation_plot_heatmaps_brain_schematic(self, DictDf, DictDf_rgba_values, 
        savedir="/tmp", DEBUG=False):
        """ Maps the outputs from heatmaps onto brain scheamtic.
        Just plot, doesnt do any computation here
        """
        import matplotlib.pyplot as plt

        # 1) DEFINE COORDS FOR EACH REGION
        # (horiz from left, vert from top)
        map_bregion_to_location = {}
        map_bregion_to_location["00_M1_m"] = [0, 1.3]
        map_bregion_to_location["01_M1_l"] = [1, 2]
        map_bregion_to_location["02_PMv_l"] = [4, 5.3]
        map_bregion_to_location["03_PMv_m"] = [3.5, 3.3]
        map_bregion_to_location["04_PMd_p"] = [3.3, 1.6]
        map_bregion_to_location["05_PMd_a"] = [5, 1.85]
        map_bregion_to_location["06_dlPFC_p"] = [7.2, 2.8]
        map_bregion_to_location["07_dlPFC_a"] = [9, 3]
        map_bregion_to_location["08_vlPFC_p"] = [5.8, 5]
        map_bregion_to_location["09_vlPFC_a"] = [8.5, 4]
        map_bregion_to_location["10_FP_p"] = [11, 3.9]
        map_bregion_to_location["11_FP_a"] = [12.5, 4.3]
        map_bregion_to_location["12_SMA_p"] = [-.1, 0.2]
        map_bregion_to_location["13_SMA_a"] = [1.4, 0.3]
        map_bregion_to_location["14_preSMA_p"] = [3.2, 0.4]
        map_bregion_to_location["15_preSMA_a"] = [4.5, 0.6]
        xmult = 33
        ymult = 50
        # xoffset = 230 # if use entire image
        xoffset = 100 # if clip
        yoffset = 30
        for k, v in map_bregion_to_location.items():
            map_bregion_to_location[k] = [xoffset + xmult*v[0], yoffset + ymult*v[1]]
        rad = (xmult + ymult)/4

        # 2) Plot all heatmaps
        list_var_heatmaps = DictDf_rgba_values.keys()

        for var in list_var_heatmaps:

            # Extract the data and rgba values
            dfthis_agg_2d = DictDf[var]
            rgba_values = DictDf_rgba_values[var]

            map_bregion_to_rowindex = {}
            list_regions = dfthis_agg_2d.index.tolist()
            for i, region in enumerate(list_regions):
                map_bregion_to_rowindex[region] = i
                
            if DEBUG:
                print("\nindex -- region")
                for k, v in map_bregion_to_rowindex.items():
                    print(v, k)

            map_event_to_colindex = {}
            list_events = dfthis_agg_2d.columns.tolist()
            for i, event in enumerate(list_events):
                map_event_to_colindex[event] = i
            if DEBUG:
                print("\nindex -- event")
                for event, i in map_event_to_colindex.items():
                    print(i, ' -- ' , event)

            # PLOT:
            ncols = 4
            nrows = int(np.ceil(len(list_events)/ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))

            for i, (ax, event) in enumerate(zip(axes.flatten(), list_events)):

                ax.set_title(event)
                ax.set_ylabel(var)
                
                # 1) load a cartoon image of brain
            #     image_name = "/home/lucast4/Downloads/thumbnail_image001.png"
                image_name = "/gorilla3/Dropbox/SCIENCE/FREIWALD_LAB/DATA/brain_drawing_template.jpg"
                im = plt.imread(image_name)
                im = im[:330, 130:]
                ax.imshow(im)

            #     if i==1:
            #         assert False
                for bregion in list_regions:
                    irow = map_bregion_to_rowindex[bregion]
                    icol = map_event_to_colindex[event]

                    col = rgba_values[irow, icol]
                    cen = map_bregion_to_location[bregion]

                    # 2) each area has a "blob", a circle on this image

                    c = plt.Circle(cen, rad, color=col, clip_on=False)
                    ax.add_patch(c)

            # SAVE FIG
            fig.savefig(f"{savedir}/brainschem-{var}.pdf")

    def modulation_plot_each_chan(self, RES_ALL_CHANS, savedir="/tmp",
        DEBUG = False, list_chans=None):
        """ Plot for each chan, overview of modulation across all variables.
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from pythonlib.tools.plottools import makeColors
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel

        def _find_varhue_varcol(var_x, vars_exist, 
                                variables_ordered_increasing_effect = None):
            """ to return what variables to use as hue and column for 
            seaborn catplot, based on principle that variables with largest
            expected effect (differences across lewvels) should be col, while
            those with smallest effect shoudl be hue (so that they are
            easy to compare)
            PARAMS:
            - variables_ordered_increasing_effect, list of var to sample
            from
            RETURNS: 
            - var_x, 
            - var_hue, 
            - var_col
            """


            if variables_ordered_increasing_effect is None:
                variables_ordered_increasing_effect = ("gridsize", "gridloc", "shape_oriented", "chunk_rank', 'chunk_within_rank",
                    "velmean_x", "velmean_y", "velmean_thbin", "velmean_norm", "velmean_normbin", "epoch")
            def _find_hue():    
                # 1) Find the hue
                for var_other in variables_ordered_increasing_effect:
                    if not var_other==var_x and var_other in vars_exist:
                        return var_other

                print(var_x, variables_ordered_increasing_effect, vars_exist)
                assert False, "didnt find"

            def _find_col(var_hue):
                # 2) Find the column
                for var_other in variables_ordered_increasing_effect:
                    if not var_other==var_x and not var_other==var_hue and var_other in vars_exist:
                        return var_other
                print(var_x, var_hue, variables_ordered_increasing_effect, vars_exist)
                assert False, "didnt find"
            
            if [var_x] == vars_exist:
                # Then no other vars exist
                return var_x, None, None
            var_hue = _find_hue()

            if sorted([var_x, var_hue]) == vars_exist:
                # Then no other vars exist
                return var_x, var_hue, None
            var_col = _find_col(var_hue)

            return var_x, var_hue, var_col

        # Prepare variables
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        sites_keep = self.Sites
        if DEBUG:
            sites_keep = sites_keep[::5]
        DF = self.DfScalar
        sn = self.SN
        list_var = self.Params["list_features_get_conjunction"]
        map_var_to_othervars = self.Params["map_var_to_othervars"]
        map_var_to_levels = self.Params["map_var_to_levels"]
        for RES_ALL in RES_ALL_CHANS:
            chan = RES_ALL["chan"]
            if list_chans is not None:
                if chan not in list_chans:
                    continue
            bregion = RES_ALL["bregion"]
            RES = RES_ALL["RES"]
            RES_FR = RES_ALL["RES_FR"]
            print("Plotting for chan: ", chan)
            dfthis = DF[DF["chan"] == chan]
            ymax = dfthis["fr_scalar"].max()

            print("PLotting for (chan, bregion): ", chan, bregion)
            
            ##################### Plot separately each var (showing its modulation)
            if len(list_var)>2:
                # otherwise doesnt make sense, this is all captured int he overview plot.
                for xvar in list_var:
                    _, var_hue, var_col = _find_varhue_varcol(xvar, list_var)
                    
                    fig = sns.catplot(data=dfthis, x=xvar, y="fr_scalar", hue=var_hue, 
                        row="event_aligned", col=var_col, kind="point", height=3)
                    rotateLabel(fig)
                
                    # fr, scale from 0
                    for ax in fig.axes.flatten():
                        ax.set_ylim([0, ymax])

                    # Save
                    fig.savefig(f"{savedir}/{bregion}-{chan}-x_{xvar}.pdf")
                      
            #################### A SINGLE OVERVIEW PLOT
            nrows = len(list_events_uniqnames)+1
            ncols = len(list_var)+1
            fig, axes = plt.subplots(nrows, ncols,  figsize=(ncols*4, nrows*4))
            
            # === 1) Overview, single plot, for each var, plot it over conjunction of other vars
            for j, var in enumerate(list_var):
                for i, ev in enumerate(list_events_uniqnames):

                    ax = axes[i][j]
                    ax.set_title(ev)
                    dfthisthis = dfthis[dfthis["event_aligned"]==ev]
                    other_vars = map_var_to_othervars[var] # conjucntion of other varss
                    g = sns.pointplot(ax=ax, data=dfthisthis, x=var, y="fr_scalar", hue=other_vars)
                    if i>0:
                        # only keep legend for first row
                        g.legend().remove()        
                            
                    # fr, scale from 0
                    ax.set_ylim([0, ymax])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

                    if j==1:
                        ax.set_ylabel(var)
                    if i==0:
                        ax.legend(framealpha=0.5)

                # also plot all combined
                ax = axes[len(list_events_uniqnames)][j]
                sns.pointplot(ax=ax, data=dfthis, x=var, y = "fr_scalar", hue="event_aligned")
                
                # fr, scale from 0
                ax.set_ylim([0, ymax])
                ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
                ax.legend(framealpha=0.5)

            # === 2) Modulation, plot across events
            ax = axes[len(list_events_uniqnames)][ncols-1]
            ax2 = axes[len(list_events_uniqnames)-1][ncols-1]
            pcols = makeColors(len(list_var))
            for var, pcol in zip(list_var, pcols):
                
                vals = RES["modulation_across_events"][var]
                ax.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)
                ax2.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)
                
                vals = RES["modulation_across_events_subgroups"][var]
                ax.plot(list_events_uniqnames, vals, '--o', color=pcol, label=f"{var}_othervars_mean")
                ax2.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)

            ax.set_ylim([0, 0.5])
            
            ax.legend(framealpha=0.5)
            ax.set_title('Modulation, across events')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
            ax2.legend(framealpha=0.5)
            ax2.set_title('Modulation, across events')
            ax2.set_xticklabels(ax.get_xticklabels(), rotation = 45)


            # == 3) Inconsistency across subgroupings
            ax = axes[len(list_events_uniqnames)-2][ncols-1]
            pcols = makeColors(len(list_var))
            for var, pcol in zip(list_var, pcols):

                # v1) Difference
                vals_diff = RES["inconsistency_across_events"][var]
                ax.plot(list_events_uniqnames, vals_diff, '-o', color=pcol, label=var)
                
        #         # v2) quotient
        #         inconsistency = 1 - vals_all/vals_sub
        #         ax2.plot(list_events_uniqnames, inconsistency, '-o', color=pcol, label=var)
            ax.legend(framealpha=0.5)
            # ax.set_ylim([0, 0.25])
            ax.set_ylabel("modulation(sub) - modulation(all)")
            ax.set_title('Inconsistency score')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

            # == 4 Modulation, all events vs. modulation, each event
            ax = axes[0][ncols-1]
            for var in list_var:    
                y_vals = RES["avgmodulation_across_methods"][var]
                x_labels = RES["avgmodulation_across_methods_labels"]
                ax.plot(x_labels, y_vals, '-o', label=var)
            ax.set_title("avg_modulation_across_methods")
            ax.legend(framealpha=0.5)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_ylim([0, 0.5])
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

            # b) mean fr across events (simple)
            ax = axes[1][ncols-1]
            sns.pointplot(ax=ax, data=dfthis, x="event_aligned", y = "fr_scalar")
            # fr, scale from 0
            ax.set_ylim([0, ymax])
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
            ax.set_title("firing rate")

            # b) Consistency
            # TODO!!! 
            
            fig.savefig(f"{savedir}/{bregion}-{chan}-overview.pdf")
            plt.close("all")
            
    def modulation_plot_summarystats(self, OUT, savedir="/tmp", which_modulation_variable="scalar"):
        """ 
        Plot many variations, for output from modulation_compute_higher_stats
        PARAMS:
        - OUT, output from modulation_compute_higher_stats
        """
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.snstools import plotgood_lineplot
        import seaborn as sns

        # Extract dataframes from OUT
        dfdat_var_events = OUT["dfdat_var_events"]
        dfdat_var_methods = OUT["dfdat_var_methods"]
        dfdat_events = OUT["dfdat_events"]
        dfdat_summary_mod = OUT["dfdat_summary_mod"]
        dfdat_fr_events = OUT["dfdat_fr_events"]
        list_var = self.Params["list_features_get_conjunction"]
        list_events_uniqnames = self.Params["list_events_uniqnames"]

        # if which_modulation_variable in ["modulation_across_events_usingsmfr", "modulation_across_events_usingsmfr_zscored"]:
        #     dfdat_var_events = dfdat_var_events.copy()
        #     from pythonlib.tools.pandastools import applyFunctionToAllRows
        #     def F(x):
        #         if x["val_kind"] == 
        #     dfdat_var_events["modulation_across_events"] = dfdat_var_events[which_modulation_variable]
        #     dfdat_var_events["modulation_across_events_subgroups"] = dfdat_var_events[which_modulation_variable]

        which_modulation_variable

        if which_modulation_variable in ["modulation_across_events_usingsmfr", "modulation_across_events_usingsmfr_zscored"]:
            MODULATION_COLS_TO_PLOT = [which_modulation_variable]
        else:
            MODULATION_COLS_TO_PLOT = ["modulation_across_events", "modulation_across_events_subgroups"]

        # Compare across events
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(MODULATION_COLS_TO_PLOT)]
        fig = sns.catplot(data=dfthis, x="event", y="val", hue="val_kind",
                          col="bregion", row="var", kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/1_lines_modulation_kinds.pdf")


        # Comparing brain regions
        for val_kind in MODULATION_COLS_TO_PLOT:
            dfthis = dfdat_var_events[dfdat_var_events["val_kind"]==val_kind]
            fig = sns.catplot(data=dfthis, x="bregion", y="val", col="event", 
                        row="var", kind="bar", aspect=2, height=4);
            rotateLabel(fig)
            fig.savefig(f"{savedir}/2_bars_feature_vs_event-{val_kind}.pdf")      

        # Comparing brain regions
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(MODULATION_COLS_TO_PLOT)]
        fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind",
                    row="event", kind="bar", hue="var", aspect=2, height=4);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/2_bars_feature_vs_valkind.pdf")


        # Compare across events
        dfthis = dfdat_var_events
        fig = sns.catplot(data=dfthis, x="event", y="val", hue="var",
                          col="bregion", row="val_kind", kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/3_lines_valkinds.pdf")

        for val_kind in MODULATION_COLS_TO_PLOT:
            dfthis = dfdat_var_events[dfdat_var_events["val_kind"]==val_kind]
            fig = plotgood_lineplot(dfthis, xval="event", yval="val", line_grouping="chan",
                                    include_mean=True, 
                                    relplot_kw={"row":"var", "col":"bregion"});
            rotateLabel(fig)
            fig.savefig(f"{savedir}/4_lineschans_feature_vs_region-{val_kind}.pdf")


        dfthis = dfdat_events
        fig = sns.catplot(data=dfthis, x="event", y="val", hue="val_kind",
                          col="bregion", col_wrap=4, kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/5_lines_fr.pdf")


        dfthis = dfdat_events
        fig = plotgood_lineplot(dfthis, xval="event", yval="val", line_grouping="chan",
                                include_mean=True, colvar="bregion", col_wrap=4);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/5_lineschans_fr.pdf")


        # AVGMOD
        dfthis = dfdat_var_methods
        fig = sns.catplot(data=dfthis, x="method", y="val", hue="var",
                          col="bregion", col_wrap=4, kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/6_lines_avgmodulation_vs_method.pdf")

        # AVGMOD
        dfthis = dfdat_var_methods
        fig = plotgood_lineplot(dfthis, xval="method", yval="val", line_grouping="chan",
                                include_mean=True, 
                                relplot_kw={"row":"var", "col":"bregion"});
        rotateLabel(fig)
        fig.savefig(f"{savedir}/6_lineschans_avgmodulation_vs_method.pdf")


        # Comparing brain regions
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(MODULATION_COLS_TO_PLOT)]
        fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind",
                    row="var", kind="bar", hue="event", aspect=2.5, height=4);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/7_bars_feature_vs_valkind.pdf")


        dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_delt" for var in list_var])]
        # fig = sns.catplot(data=dfthis, x="val_kind", y="val", col="bregion", col_wrap=4, aspect=1.5, height=2,
        #                  kind="bar");
        fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind", aspect=2, height=4, kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_deltsonly.pdf")

        dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_mean" for var in list_var])]
        fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind", aspect=2, height=4, kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_meansonly.pdf")


        dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_delt" for var in list_var])]
        fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
                         kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_features_deltonly.pdf")

        dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_mean" for var in list_var])]
        fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
                         kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_features_meanonly.pdf")


        dfthis = dfdat_summary_mod
        fig = sns.catplot(data=dfthis, x="val_kind", y="val", col="bregion", col_wrap=4, aspect=1.5, height=2,
                         kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_all.pdf")
        # fig = sns.catplot(data=dfthis, x="bregion", y="val", hue="val_kind",aspect=2, height=4,
        #                  kind="bar");
        # from pythonlib.tools.snstools import rotateLabel
        # rotateLabel(fig)

        fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
                         kind="bar");
        rotateLabel(fig)
        fig.savefig(f"{savedir}/8_bars_features_all.pdf")

        ################## FR MODULATION BY LEVELS
        if len(dfdat_fr_events)>0:
            list_val_kind = dfdat_fr_events["val_kind"].unique().tolist()
            for var in list_var:
                for val_kind in list_val_kind:
                    dfthis = dfdat_fr_events[(dfdat_fr_events["var"]==var) & (dfdat_fr_events["val_kind"]==val_kind)]

                    if len(dfthis)>0:
                        fig = sns.catplot(data=dfthis, x="var_level", hue="event", y="val", col="bregion", col_wrap=4, kind="point")
                        rotateLabel(fig)
                        fig.savefig(f"{savedir}/9_{var}-{val_kind}-lines_fr_vs_level.pdf")

                        try:
                            fig = plotgood_lineplot(dfthis, xval="var_level", yval="val", line_grouping="chan",
                                                    include_mean=True, 
                                                    relplot_kw={"row":"event", "col":"bregion"});
                            rotateLabel(fig)
                            fig.savefig(f"{savedir}/9_{var}-{val_kind}-lineschans_fr_vs_level.pdf")
                        except Exception as err:
                            print(dfthis["var_level"].value_counts())
                            print(dfthis["chan"].value_counts())
                            print(dfthis["event"].value_counts())
                            print(dfthis["bregion"].value_counts())
                            # raise err
                            fig = None

    def modulation_plot_all(self, RES_ALL_CHANS, OUT, SAVEDIR, 
            list_plots = ("summarystats", "heatmaps", "eachsite_allvars", 
                "eachsite_smfr", "eachsite_rasters"), 
            suffix=None, list_sites=None,
            which_modulation_variable = "scalar",
            which_var_heatmaps = "modulation_across_events_subgroups"):
        """ Plot all summary plots for this dataset (self)
        PARAMS;
        - list_plots, list of str, to plot
        - RES_ALL_CHANS, optional, output of self.modulation_compute_each_chan
        - OUT, optional, output of self.modulation_compute_higher_stats
        """

        # # Get subset of sites
        # sitesall = self.SN.sitegetter_all()
        # list_sites = self.SN._sitegetter_sort_sites_by(sitesall, "fr", take_top_n=20)

        import os
        def _finalize_dir(savedir):
            if suffix is not None:
                savedir = f"{savedir}-{suffix}"
            os.makedirs(savedir, exist_ok=True)
            return savedir

        for plotkind in list_plots:
            if plotkind=="summarystats":
                # Summary over all chans for each site
                savedir = _finalize_dir(f"{SAVEDIR}/modulation_by_features-{which_modulation_variable}")
                print(f"Plotting {plotkind} at: {savedir}")
                self.modulation_plot_summarystats(OUT, savedir=savedir,
                    which_modulation_variable=which_modulation_variable)
            elif plotkind=="heatmaps":
                # Plot heatmaps and brain schematics
                savedir = _finalize_dir(f"{SAVEDIR}/modulation_heatmaps-{which_var_heatmaps}")
                print(f"Plotting {plotkind} at: {savedir}")
                DictDf, DictDf_rgba_values = self.modulation_plot_heatmaps(OUT, 
                    savedir=savedir, val_kind=which_var_heatmaps)
                self.modulation_plot_heatmaps_brain_schematic(DictDf, DictDf_rgba_values, savedir)
            elif plotkind=="eachsite_allvars":
                # Plot overview for each channel
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_all_vars")
                print(f"Plotting {plotkind} at: {savedir}")
                os.makedirs(savedir, exist_ok=True)
                self.modulation_plot_each_chan(RES_ALL_CHANS, savedir, list_chans=list_sites)
            elif plotkind=="eachsite_smfr":
                # Plot smoothed fr for each channel
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_smoothedfr")
                os.makedirs(savedir, exist_ok=True)
                print(f"Plotting {plotkind} at: {savedir}")
                if list_sites is None:
                    list_sites_this = self.Sites
                else:
                    list_sites_this = list_sites
                for site in list_sites_this:
                    # Smoothed FR (average for each level)
                    self.plot_smfr_average_each_level(site, savedir);
                    plt.close("all")
            elif plotkind=="eachsite_smfr_eachtrial":
                # Plot smoothed fr for each channel
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_smoothedfr")
                os.makedirs(savedir, exist_ok=True)
                print(f"Plotting {plotkind} at: {savedir}")
                if list_sites is None:
                    list_sites_this = self.Sites
                else:
                    list_sites_this = list_sites
                for site in list_sites_this:
                    # Plot smoothed fr (each trial)
                    self.plot_smfr_trials_each_level(site, savedir, alpha=0.3);
                    plt.close("all")
            elif plotkind=="eachsite_smfr_splitby_character":
                # Plot smoothed fr for each channel
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_smoothedfr_splitby_character")
                os.makedirs(savedir, exist_ok=True)
                print(f"Plotting {plotkind} at: {savedir}")
                if list_sites is None:
                    list_sites_this = self.Sites
                else:
                    list_sites_this = list_sites
                for site in list_sites_this:
                    # Smoothed FR (average for each level)
                    self.plot_smfr_average_each_level(site, savedir, list_var=["character"]);
                    
                    # # Plot smoothed fr (each trial)
                    # self.plot_smfr_trials_each_level(site, savedir, alpha=0.3);
                    plt.close("all")
            elif plotkind=="eachsite_rasters":
                # Plot Rasters for each channel
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_rasters")
                os.makedirs(savedir, exist_ok=True)    
                print(f"Plotting {plotkind} at: {savedir}")
                self.plot_rasters_split_by_feature_levels(list_sites, savedir)
            elif plotkind=="eachsite_raster_splitby_character":
                # each characters is a single group
                savedir = _finalize_dir(f"{SAVEDIR}/each_chan_rasters_splitby_character")
                os.makedirs(savedir, exist_ok=True)    
                print(f"Plotting {plotkind} at: {savedir}")
                self.plot_rasters_split_by_feature_levels(list_sites, savedir,
                    list_var = ["character"])
            else:
                print(plotkind)
                assert False
            plt.close("all")



    ############### PLOTS
    def plotmod_overlay_event_boundaries(self, ax, event, overlay_pre_and_post_boundaries=True):
        """ Overlay the boundaries of this event (vertical lines)
        """

        pre_dur, post_dur = self.event_extract_pre_post_dur(event)
        # Overlay event bounds
        if overlay_pre_and_post_boundaries:
            event_bounds = [pre_dur, 0., post_dur]
        else:
            event_bounds = [None, 0., None]

        colors = ['r', 'k', 'b']
        for evtime, pcol in zip(event_bounds, colors):
            if evtime is not None:
                ax.axvline(evtime, color=pcol, linestyle="--", alpha=0.4)


    ########################################
    def plotwrapper_heatmap_smfr(self, which_var = "event_aligned", sdir=None):
        """ Plot smoothed, average FR in heatmap, one figure for each region,
        one subplot for each event, and each unit a row in this subplot
        """

        sn = self.SN
        list_event_aligned = self.DfScalar[which_var].unique().tolist()

        if which_var!="event_aligned":
            event0 = self.Params["list_events_uniqnames"][0]
            assert len(self.Params["list_events_uniqnames"])==1, "multipoel events ,not sure what to use for timing..."

        # if which_var=="event_aligned":
        #     list_event_aligned = self.Params["list_events_uniqnames"]
        # else:
        #     list_event_aligned = self.Params["map_var_to_levels"][which_var]
        #     event0 = self.Params["list_events_uniqnames"][0]
        #     assert len(self.Params["list_events_uniqnames"])==1, "multipoel events ,not sure what to use for timing..."


        ZSCORE = True
        list_regions = sn.sitegetter_get_brainregion_list()
        zlims = [-2, 2]
        # zlims = [None, None]

        for region in list_regions:
            print("Plotting...", region)
            sites_this = [s for s in self.Sites if s in sn.sitegetter_map_region_to_sites(region)]

            # 1) extract smoothed FR for each  unit, for each event
            fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*4, 8))
            
            # 2) Collect all fr mat
            List_fr_mat = []
            for i, (event, ax) in enumerate(zip(list_event_aligned, axes.flatten())):
                            
                # extract matrix of mean fr (site x time)
                list_frmean = []
                for site in sites_this:
                    frmat = self.dataextract_as_frmat(site, var=which_var, 
                        var_level=event, return_as_zscore=ZSCORE) 
                    frmean = np.mean(frmat, 0)
                    list_frmean.append(frmean)

                frmat_site_by_time = np.stack(list_frmean)

                # sort (only for first index)
                if i==0:
                    frmat_site_by_time, sites_this = self.frmat_sort_by_time_of_max_fr(frmat_site_by_time, 
                        sites_this)

                # normalize firing rates (use percent change from first time bin)
                if False:
                    frmedian = np.median(frmat_site_by_time, axis=1, keepdims=True)
                    frmat_site_by_time = (frmat_site_by_time - frmedian)/frmedian

                List_fr_mat.append(frmat_site_by_time)

            # Figure out zlim
            zmax = np.max(np.abs([x.max() for x in List_fr_mat] + [x.min() for x in List_fr_mat]))
            if zmax>1.8:
                zmax = 1.8
            
            for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
                if which_var=="event_aligned":
                    times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event)
                else:
                    times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event0)

                # plot as 2d heat map
                from pythonlib.tools.snstools import heatmap_mat
            #         sns.heatmap(frmat_site_by_time, ax=ax, )
                _, ax, _ = heatmap_mat(frmat_site_by_time, annotate_heatmap=False, ax=ax, diverge=True, zlims=[-zmax, zmax]);

                # Set y and x ticks
                ax.set_yticks([i+0.5 for i in range(len(sites_this))], labels=sites_this)
                ax.set_xticks(xticks, labels=xticklabels)
                ax.axvline(ind_0)
                
                ax.set_title(event)
            
            fig.savefig(f"{sdir}/zscored-{region}.pdf")
            
            # plot the means
            fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*3, 3))
            from neuralmonkey.neuralplots.population import plotNeurTimecourse, plot_smoothed_fr

            for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
                if which_var=="event_aligned":
                    times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event)
                else:
                    times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event0)

                plot_smoothed_fr(frmat_site_by_time, times, ax=ax)
                ax.axvline(times[ind_0])
                ax.set_title(event)
                    
            if sdir:
                fig.savefig(f"{sdir}/zscored-{region}-mean.pdf")
            
            plt.close("all")        


    def plot_smfr_trials_each_level(self, chan, savedir=None, 
        alpha=0.2):
        """" Plot smoothed fr, one curve for each trial, split into supblots, one
        for each combo of (event, var, level). Plots all of those combos.
        """
        
        # smoothed fr, trials, each level
        bregion = self.SN.sitegetter_thissite_info(chan)["region"]
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        list_pre_dur = self.Params["list_pre_dur"]
        list_post_dur = self.Params["list_post_dur"]
        map_var_to_levels = self.Params["map_var_to_levels"]
        list_var = self.Params["list_features_get_conjunction"]

        for var in list_var:
            list_levels = map_var_to_levels[var]
            
            nrows = len(list_events_uniqnames)
            ncols = len(list_levels)
            fig, axes = plt.subplots(nrows, ncols, 
                sharex="row", sharey=True,
                figsize=(ncols*1.75, nrows*1.3))

            for i, event in enumerate(list_events_uniqnames):
                for j, lev in enumerate(list_levels):

                    ax = axes[i][j]

                    if i==0:
                        ax.set_title(lev)
                    if j==0:
                        ax.set_ylabel(event)

                    # for each (chan, level, plot smoothed fr)

                    # Get for this event
                    pathis = self.popanal_extract_specific_slice(event, chan, (var, lev))

                    # Plot it
                    pathis.plotwrapper_smoothed_fr(ax=ax, plot_indiv=True, 
                        plot_summary=True, alpha=alpha)
            
            if savedir is not None:
                fig.savefig(f"{savedir}/{bregion}-{chan}-smfr_trials-{var}.pdf")

        return fig, axes



    def plot_smfr_average_each_level(self, chan, savedir=None,
        list_var = None, list_events_uniqnames=None, orient="vert",
        width=3, height=2, overlay_pre_and_post_boundaries = True):
        """ For each var a subplot, overlaying smoothed fr for each level for that
        var. Also splits by event.
        """

        list_events_uniqnames, _, list_pre_dur, list_post_dur = \
            self.event_list_extract_linked_params(list_events_uniqnames)

        # Info
        bregion = self.SN.sitegetter_thissite_info(chan)["region"]
        # if list_events_uniqnames is None:
        #     list_events_uniqnames = self.Params["list_events_uniqnames"]
        # list_pre_dur = self.Params["list_pre_dur"]
        # list_post_dur = self.Params["list_post_dur"]
        if list_var is None:
            list_var = self.Params["list_features_get_conjunction"]
        map_var_to_levels = self.Params["map_var_to_levels"]
        for var in list_var:
            if var not in map_var_to_levels.keys():
                self._preprocess_map_features_to_levels_input(var)

        if orient=="vert":
            nrows = len(list_events_uniqnames)
            ncols = len(list_var)
            share = "row"
        elif orient=="horiz":
            nrows = len(list_var)
            ncols = len(list_events_uniqnames)
            share="col"
        else:
            assert False
        fig, axes = plt.subplots(nrows, ncols, sharex=share, sharey=True, 
            figsize=(ncols*width, nrows*height), squeeze=False)

        for i, (event, pre_dur, post_dur) in enumerate(zip(list_events_uniqnames, list_pre_dur, list_post_dur)):
            for k, var in enumerate(list_var):
                
                if orient=="vert":
                    ax = axes[i][k]
                elif orient=="horiz":
                    ax = axes[k][i]
                else:
                    assert False

                if overlay_pre_and_post_boundaries:
                    event_bounds=[pre_dur, 0., post_dur]
                else:
                    event_bounds=[None, 0., None]

                # Each level is a single smoothed fr, diff color
                list_levels = map_var_to_levels[var]
                                    
                # Get for this event
                pathis = self.popanal_extract_specific_slice(event, chan)

                # Plot
                add_legend=i==0
                pathis.plotwrapper_smoothed_fr_split_by_label("trials", var, 
                    ax=ax, event_bounds=event_bounds,
                    add_legend=add_legend)

                if i==0:
                    ax.set_title(var)
                    # ax.legend(list_levels, framealpha=0.4)
                if k==0:
                    ax.set_ylabel(event)

        if savedir is not None:
            fig.savefig(f"{savedir}/{bregion}-{chan}-smfr_avg_allvars.pdf")

        return fig, axes
                    

    def plot_rasters_split_by_feature_levels(self, list_sites=None, savedir=None,
            list_var = None, save_ext="png", list_events=None, orient="vert"):
        """ Plot each site, and also split into each feature and event.
        BEtter to plot separately becuase crashes if try to have them all as
        separeat subplots
        """

        if list_sites is None:
            list_sites = self.Sites
        if list_events is None:
            list_events = self.Params["list_events_uniqnames"]
        if list_var is None:
            list_var = self.Params["list_features_get_conjunction"]

        for site in list_sites:
            bregion = self.SN.sitegetter_thissite_info(site)["region"]

            for var in list_var:
                fig, axes = self._plot_rasters_split_by_feature_levels(site, 
                    [var], list_events_uniqnames=list_events, orient=orient)
                if savedir is not None:
                    # fig.savefig(f"{savedir}/{bregion}-{site}-{var}.{save_ext}", dpi=300)
                    fig.savefig(f"{savedir}/{bregion}-{site}-{var}.{save_ext}")

                # OLD: split by event.
                # for ev in list_events:
                #     fig, axes = self._plot_rasters_split_by_feature_levels(site, 
                #         [var], [ev])

                #     fig.savefig(f"{savedir}/{bregion}-{site}-{var}-{ev}.png")

                plt.close("all")



    def _plot_rasters_split_by_feature_levels(self, site, 
        list_var = None, list_events_uniqnames = None, orient="vert",
        width = 4, height = 3, overlay_pre_and_post_boundaries = True):
        """ Plot rasters, comparing all trials across levels for each var and event 
        combo
        """
        
        # 1) Extract the trials in SN which correspond to each of the levels 
        # for this variable(feature).        
        if list_var is None:
            list_var = self.Params["list_features_get_conjunction"]
        map_var_to_levels = self.Params["map_var_to_levels"]
        # overlay event boundaires

        # # same length lists, len num events.
        # list_events_uniqnames_ALL = self.Params["list_events_uniqnames"]
        # list_events_orig_ALL = self.Params["list_events"]
        # list_pre_dur_ALL = self.Params["list_pre_dur"]
        # list_post_dur_ALL = self.Params["list_post_dur"]

        # if list_events_uniqnames is not None:
        #     # Pull out these specific events, and assopcaited params
        #     list_idx = [list_events_uniqnames_ALL.index(ev) for ev in list_events_uniqnames] 

        #     list_events_uniqnames = [list_events_uniqnames_ALL[i] for i in list_idx]
        #     list_events_orig = [list_events_orig_ALL[i] for i in list_idx]
        #     list_pre_dur = [list_pre_dur_ALL[i] for i in list_idx]
        #     list_post_dur = [list_post_dur_ALL[i] for i in list_idx]
        # else:
        #     list_events_uniqnames = list_events_uniqnames_ALL
        #     list_events_orig = list_events_orig_ALL
        #     list_pre_dur = list_pre_dur_ALL
        #     list_post_dur = list_post_dur_ALL

        list_events_uniqnames, list_events_orig, list_pre_dur, list_post_dur = self.event_list_extract_linked_params(list_events_uniqnames)

        if orient=="vert":
            ncols = len(list_var)
            nrows = len(list_events_uniqnames)
        elif orient=="horiz":
            ncols = len(list_events_uniqnames)
            nrows = len(list_var)
        else:
            print(orient)
            assert False

        fig, axes = plt.subplots(nrows, ncols, squeeze=False, 
            figsize=(ncols*width, nrows*height))

        for i, var in enumerate(list_var):
            for j, (event, event_orig, pre_dur, post_dur) in \
                enumerate(zip(list_events_uniqnames, list_events_orig, list_pre_dur, list_post_dur)):

                if orient=="vert":
                    ax = axes[j][i]
                elif orient=="horiz":
                    ax = axes[i][j]
                else:
                    assert False

                if var not in map_var_to_levels.keys():
                    self._preprocess_map_features_to_levels_input(var)
                list_levels = map_var_to_levels[var]
                
                # collect trials in the order you want to plot them (bottom to top)
                # get trialscodes from SP
                list_trials_sn = []
                list_labels = []
                for lev in list_levels:
                    pathis = self.popanal_extract_specific_slice(event, site, (var, lev)) 
                    
                    # get the original trialcodes
                    list_trialcode = pathis.Xlabels["trials"]["trialcode"]

                    # map them to trials in sn
                    trials_sn = self.SN.datasetbeh_trialcode_to_trial_batch(list_trialcode)
                    if len(trials_sn)>0:
                        list_trials_sn.append(trials_sn)
                        list_labels.append(lev)

                # method in sn, plitting rasters with blocked trials
                self.SN.plot_raster_trials_blocked(ax, list_trials_sn, site, list_labels, 
                                           alignto=event_orig,
                                           overlay_trial_events=False, xmin=pre_dur-0.2, 
                                           xmax=post_dur+0.2)

                self.plotmod_overlay_event_boundaries(ax, event, overlay_pre_and_post_boundaries=overlay_pre_and_post_boundaries)


                if j==0:
                    ax.set_title(var)
                if i==0:
                    ax.set_ylabel(event_orig)
        return fig, axes


    def plot_rasters_smfr_each_level_combined(self, site, var, 
        list_events_uniqnames = None, orient="vert",
        width = 4, height = 3, overlay_pre_and_post_boundaries = True):
        """ [Good], plot in a single figure both rasters and sm fr, aligned.
        e.g, if orient is "horiz", then each col is an event, each row
        is rasters (top) and sm fr (bottom).
        PARAMS;
        - var, str, the variable, e.g, epoch"
        - width, height, for each subplot.
        - overlay_pre_and_post_boundaries, if false, then just overlays the zero-time
        """
        
        # 1) Extract the trials in SN which correspond to each of the levels 
        # for this variable(feature).        
        map_var_to_levels = self.Params["map_var_to_levels"]
        list_events_uniqnames, list_events, list_pre_dur, list_post_dur = self.event_list_extract_linked_params(list_events_uniqnames)

        if orient=="vert":
            ncols = 2
            nrows = len(list_events_uniqnames)
            sharex = "row"
            sharey = "col"
        elif orient=="horiz":
            ncols = len(list_events_uniqnames)
            nrows = 2
            sharex = "col"
            sharey = "row"
        else:
            print(orient)
            assert False

        fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey,
            figsize=(ncols*width, nrows*height))

        for j, (event, event_orig, pre_dur, post_dur) in \
            enumerate(zip(list_events_uniqnames, list_events, list_pre_dur, list_post_dur)):

            if orient=="vert":
                ax1 = axes[j][0]
                ax2 = axes[j][1]
            elif orient=="horiz":
                ax1 = axes[0][j]
                ax2 = axes[1][j]
            else:
                assert False

            if var not in map_var_to_levels.keys():
                self._preprocess_map_features_to_levels_input(var)
            list_levels = map_var_to_levels[var]
            
            ################# PLOT RASTER
            # collect trials in the order you want to plot them (bottom to top)
            # get trialscodes from SP
            list_trials_sn = []
            list_labels = []
            for lev in list_levels:
                pathis = self.popanal_extract_specific_slice(event, site, (var, lev)) 
                
                # get the original trialcodes
                list_trialcode = pathis.Xlabels["trials"]["trialcode"]

                # map them to trials in sn
                trials_sn = self.SN.datasetbeh_trialcode_to_trial_batch(list_trialcode)
                if len(trials_sn)>0:
                    list_trials_sn.append(trials_sn)
                    list_labels.append(lev)

            # method in sn, plitting rasters with blocked trials
            self.SN.plot_raster_trials_blocked(ax1, list_trials_sn, site, list_labels, 
                                       align_to=event_orig,
                                       overlay_trial_events=False, xmin=pre_dur-0.2, 
                                       xmax=post_dur+0.2)
            self.plotmod_overlay_event_boundaries(ax1, event, overlay_pre_and_post_boundaries=overlay_pre_and_post_boundaries)

            ################### PLOT SM FR
            # Each level is a single smoothed fr, diff color
            list_levels = map_var_to_levels[var]
            if overlay_pre_and_post_boundaries:
                event_bounds=[pre_dur, 0., post_dur]
            else:
                event_bounds=[None, 0., None]
         
            # Get for this event
            pathis = self.popanal_extract_specific_slice(event, site)

            # Plot
            # add_legend=i==0
            add_legend=j==0
            pathis.plotwrapper_smoothed_fr_split_by_label("trials", var, 
                ax=ax2, event_bounds=event_bounds,
                add_legend=add_legend)

            # make sm fr have same xlim as rasters
            ax2.set_xlim(ax1.get_xlim())

            if orient=="vert":
                ax1.set_ylabel(event)
            elif orient=="horiz":
                ax1.set_title(event)

        return fig, axes

    ########### UTILS
    def event_list_extract_linked_params(self, list_events_uniqnames=None):
        """ Helper to get things like predur etc for each even in list_events_uniqnames
        Useful for plotting, eg. when plot just subset
        PARAMS:
        - list_events_uniqnames, list of str. if None, then returns all events
        RETURNS:
        - list_events_uniqnames, list_events_orig, list_pre_dur, list_post_dur, each
        a list, matching indicesa across the lists.
        """

        list_events_uniqnames_ALL = self.Params["list_events_uniqnames"]
        list_events_orig_ALL = self.Params["list_events"]
        list_pre_dur_ALL = self.Params["list_pre_dur"]
        list_post_dur_ALL = self.Params["list_post_dur"]

        if list_events_uniqnames is not None:
            # Pull out these specific events, and assopcaited params
            list_idx = [list_events_uniqnames_ALL.index(ev) for ev in list_events_uniqnames] 

            list_events_uniqnames = [list_events_uniqnames_ALL[i] for i in list_idx]
            list_events_orig = [list_events_orig_ALL[i] for i in list_idx]
            list_pre_dur = [list_pre_dur_ALL[i] for i in list_idx]
            list_post_dur = [list_post_dur_ALL[i] for i in list_idx]
        else:
            list_events_uniqnames = list_events_uniqnames_ALL
            list_events_orig = list_events_orig_ALL
            list_pre_dur = list_pre_dur_ALL
            list_post_dur = list_post_dur_ALL

        return list_events_uniqnames, list_events_orig, list_pre_dur, list_post_dur


    def event_extract_timebins_this_event(self, event):
        """ Extract array of times for this event (sec, relative to alignement)
        PARAMS:
        - event, e.g. 00_fix_touch
        RETURNS:
        - np array, num time bins.
        """
        self.event_extract_pre_post_dur(event) # [predur, postdur]
        pa = self.popanal_extract_specific_slice(event)
        return np.array(pa.Times)

    def event_extract_time_labels(self, event):
        """
        Return labels that you can use for plotting.
        RETURNS:
        - times, np array of bins
        - ind_0, index in times that is the tiem clsoest to 0 (alignemnet)
        - xticks, [xmin, alignemnet, xmax]
        - xticklabels, strings of times, matching xticsk
        """
        times = self.event_extract_timebins_this_event(event)
        ind_0 = np.argmin(np.abs(times - 0))
        xticks = [0, ind_0, len(times)-1]
        xticklabels = [f"{times[i]:.2f}" for i in xticks]
        return times, ind_0, xticks, xticklabels


    def event_extract_pre_post_dur(self, event):
        """
        PARAMS:
        - event, unique string name (prefix num)
        RETURNS:
        - pre_dur, num
        - post_dur, num
        """

        list_events_uniqnames = self.Params["list_events_uniqnames"]
        list_pre_dur = self.Params["list_pre_dur"]
        list_post_dur = self.Params["list_post_dur"]

        ind = list_events_uniqnames.index(event)

        return list_pre_dur[ind], list_post_dur[ind]


    # def save(self, sdir, fname="Snippets", add_tstamp=True, exclude_sn=True):
    #     """ Saves self in directory sdir
    #     as pkl files
    #     """
    #     import pickle as pkl
    #     if exclude_sn:
    #         assert False, "not coded"
    #     if add_tstamp:
    #         from pythonlib.tools.expttools import makeTimeStamp
    #         ts = makeTimeStamp()
    #         fname = f"{sdir}/{fname}-{ts}.pkl"
    #     else:
    #         fname = f"{sdir}/{fname}.pkl"

    #     with open(fname, "wb") as f:
    #         pkl.dump(self, f)

    #     print(f"Saved self to {fname}")

    def save(self, savedir, name="SP.pkl"):
        """ Helper to save, which if tried to save all , would
        be >1GB, here saves ewverythign except data in SN, and 
        itmes in list_attr, which are larger data.
        SAVES to f"{savedir}/{name}"
        """
        import pickle
        
        # Temporarily store varible that will remove
        store_attr = {}
        list_attr = ["DatSpikeWaveforms", "Datasetbeh", "PopAnalDict", "DatAll", "DatAllDf",
            "DatSpikes", "DatTank"] # large size, >few hundred MB,
        for attr in list_attr:
            store_attr[attr] = getattr(self.SN, attr)
            setattr(self.SN, attr, None)
        
        # DatAll = self.SN.DatAll
        # DatAllDf = self.SN.DatAllDf
        # DatSpikes = self.SN.DatSpikes
        # DatTank = self.SN.DatTank
        # # ListPA = self.ListPA

        path = f"{savedir}/{name}"
        print("SAving: ", path)
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as err:
            # restroe this before throwing erro
            for attr in list_attr:
                setattr(self.SN, attr, store_attr[attr])

            # self.SN.DatAll = DatAll
            # self.SN.DatAllDf = DatAllDf
            # self.SN.DatSpikes = DatSpikes
            # self.SN.DatTank = DatTank
            # self.ListPA = ListPA
            raise err

        # self.SN.DatAll = DatAll
        # self.SN.DatAllDf = DatAllDf
        # self.SN.DatSpikes = DatSpikes
        # self.SN.DatTank = DatTank
        for attr in list_attr:
            setattr(self.SN, attr, store_attr[attr])

        # self.ListPA = ListPA

    def copy(self, minimal=True):
        """ make a copy, pruning variables that are not needed.
        This keeps eveyrthing except original SN'data, and listpa.
        THis allows to save without being too large (>1GB)
        PARAMS:
        - minimal, bool, if True, then copies only the PA objects
        the rest uses reference
        """

        assert False, "have to allow initializing without passing in var"
        vars_to_copy = ["ListPA", "PAscalar"]
        # Params
        # Sites
        # Trials


def extraction_helper(SN, which_level="trial", list_features_modulation_append=None,
    dataset_pruned_for_trial_analysis = None):
    """ Helper to extract Snippets for this session
    PARAMS;
    - list_features_modulation_append, eitehr None (do nothing) or list of str,
    which are features to add for computing modulation for.
    """

    # === DEFAULTS
    PRE_DUR = -0.4
    POST_DUR = 0.4

    # General cleanup of dataset.
    if dataset_pruned_for_trial_analysis is None:
        # This is generally what you want
        dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(SN)

    if which_level=="trial":
        # Events
        do_prune_by_inter_event_times = False
        dict_events_bounds, dict_int_bounds = SN.eventsanaly_helper_pre_postdur_for_analysis(
            do_prune_by_inter_event_times=do_prune_by_inter_event_times)

        # list_events = ["samp", "go", "done_button", "off_stroke_last", "reward_all"]
        list_events = list(dict_events_bounds.keys())
        list_features_extraction = []
        list_features_get_conjunction = []
        if False:
            # hard to do shuffling across events
            list_pre_dur = [dict_events_bounds[ev][0] for ev in list_events]
            list_post_dur = [dict_events_bounds[ev][1] for ev in list_events]
        else:
            list_pre_dur = [PRE_DUR for ev in list_events]
            list_post_dur = [POST_DUR for ev in list_events]
        # list_pre_dur = [-0.8 for _ in range(len(list_events))]
        # list_post_dur = [0.8 for _ in range(len(list_events))]
        strokes_only_keep_single = False
        prune_feature_levels_min_n_trials = 1

    elif which_level=="stroke":
        # Extracts snippets aligned to strokes. Features are columns in DS.
        which_level = "stroke"
        list_events = [] # must be empty
        list_features_extraction = ["shape_oriented", "gridloc"] # e.g, ['shape_oriented', 'gridloc']
        # list_features_get_conjunction = ["shape_oriented", "gridloc"] # not required, but useful if you want to get conjunction of these vars.
        list_features_get_conjunction = ["shape_oriented", "gridloc"] # REQUIRED. these are if you want to get conjunction of these vars, and
        # also those for computing moduulation.
        list_pre_dur = [PRE_DUR]
        list_post_dur = [POST_DUR]
        strokes_only_keep_single = False # if True, then prunes dataset, removing trials "remove_if_multiple_behstrokes_per_taskstroke"
        prune_feature_levels_min_n_trials = 1 
    else:
        print(which_level)
        assert False

    if list_features_modulation_append is not None:
        assert isinstance(list_features_modulation_append, list)
        list_features_extraction = list_features_extraction + list_features_modulation_append
        list_features_get_conjunction = list_features_get_conjunction + list_features_modulation_append


    SP = Snippets(SN=SN, which_level=which_level, list_events=list_events, 
                list_features_extraction=list_features_extraction, list_features_get_conjunction=list_features_get_conjunction, 
                list_pre_dur=list_pre_dur, list_post_dur=list_post_dur,
                strokes_only_keep_single=strokes_only_keep_single,
                  prune_feature_levels_min_n_trials=prune_feature_levels_min_n_trials,
                  dataset_pruned_for_trial_analysis = dataset_pruned_for_trial_analysis
                 )

    return SP


def datasetstrokes_extract(D, strokes_only_keep_single=False, tasks_only_keep_these=None, 
    prune_feature_levels_min_n_trials=None, list_features=None, vel_onset_twindow = (0, 0.2)):
    """ Helper to extract dataset strokes
    PARAMS:
    - strokes_only_keep_single, bool, if True, then prunes dataset: 
    "remove_if_multiple_behstrokes_per_taskstroke"

    """

    # 1. Extract all strokes, as bag of strokes.
    from pythonlib.dataset.dataset_strokes import DatStrokes
    DS = DatStrokes(D)
    for f in list_features:
        assert f in DS.Dat.columns, "must extract this feature first. is it a datseg feature that you failed to extract?"

    if strokes_only_keep_single:
        DS.clean_data(["remove_if_multiple_behstrokes_per_taskstroke"])
    
    if tasks_only_keep_these is not None:
        assert isinstance(tasks_only_keep_these, list)
        # DS.Dat = DS.Dat[DS.Dat["task_kind"].isin(["prims_single", "prims_on_grid"])].reset_index(drop=True)
        DS.Dat = DS.Dat[DS.Dat["task_kind"].isin(tasks_only_keep_these)].reset_index(drop=True)

    # Extract motor features, if desired.
    # list_vel_names = ["velmean_x", "velmean_y"]
    for feat in list_features:
        if "velmean_" in feat:
            DS.features_compute_velocity_binned(twind=vel_onset_twindow)
            break

    if prune_feature_levels_min_n_trials is not None:
        assert list_features is not None
        # 1) Only keep levels that have enough trials
        from pythonlib.tools.pandastools import filter_by_min_n
        assert isinstance(prune_feature_levels_min_n_trials, int)
        for var in list_features:
            print("====", var)
            DS.Dat = filter_by_min_n(DS.Dat, var, prune_feature_levels_min_n_trials)

    # Extract timing inforamtion (e.g., stroek onsets, offsets)
    DS.timing_extract_basic()


    # list_features = ["task_kind", "gridsize", "shape_oriented", "gridloc"]
    for key in list_features:
        print(" ")
        print("--- Unique levels for this category: ", key)
        print(DS.Dat[key].value_counts())

    return DS


def _dataset_extract_prune_general(sn, list_superv_keep = None):
    """
    Generic, should add to this.
    """
    Dcopy = sn.Datasetbeh.copy()

    # Remove if aborted
    Dcopy.filterPandas({"aborted":[False]}, "modify")

    if list_superv_keep is None:
        print("############ TAKING ONLY NO SUPERVISION TRIALS")
        list_superv_keep = ["off|0||0", "off|1|solid|0", "off|1|rank|0"]
    else:
        print("############ TAKING ONLY THESE SUPERVISION TRIALS:")
        print(list_superv_keep)

    # Only during no-supervision blocks
    print("--BEFORE REMOVE; existing supervision_stage_concise:")
    print(Dcopy.Dat["supervision_stage_concise"].value_counts())
    Dcopy.filterPandas({"supervision_stage_concise":list_superv_keep}, "modify")
    print("--AFTER REMOVE; existing supervision_stage_concise:")
    print(Dcopy.Dat["supervision_stage_concise"].value_counts())
    
    print("Dataset final len:", len(Dcopy.Dat))

    return Dcopy

def _dataset_extract_prune_sequence(sn, n_strok_max = 2):
    """ Prep beh dataset before extracting snippets.
    Add columns that are necessary and
    return pruned datsaet for sequence analysis.
    """

    import pandas as pd
    D = sn.Datasetbeh

#     # 1) Convert to dataset strokes (task variant)
#     from pythonlib.dataset.dataset_strokes import DatStrokes
#     # DSbeh = DatStrokes(D, version="beh")
#     DStask = DatStrokes(D, version="task") 

#     # 1) only prims in grid
#     DStask.dataset_append_column("supervision_stage_concise")
#     filtdict = {"task_kind":["prims_on_grid"], "supervision_stage_concise":["off|1|solid|0"]}
#     DStask.filter_dataframe(filtdict, True)


    # Method 1 - level of trial
    # for each trial, extract 
    datall = []
    list_dat = []
    for i in range(len(D.Dat)):

        dat = {}
        # trial-level info
        # trialcode = D.Dat.iloc[i]["trialcode"]   
        tokens_behft = D.taskclass_tokens_extract_wrapper(i)
        if False:
            dat["nstrokes_beh"] = len(D.Dat.iloc[i]["strokes_beh"])
        else:
            # Better, beucase there can be mismatch sometimes.
            dat["nstrokes_beh"] = len(tokens_behft)

        dat["nstrokes_task"] = len(D.Dat.iloc[i]["strokes_task"])

        # shapes in order
        for i in range(n_strok_max):
            if i<len(tokens_behft):
                tok = tokens_behft[i]
                dat[f"{i}_shape"] = tok["shape"]
                dat[f"{i}_loc"] = tok["gridloc"]
            else:
                dat[f"{i}_shape"] = None
                dat[f"{i}_loc"] = None

        list_dat.append(dat)

    # Put back into D.Dat
    dfdat = pd.DataFrame(list_dat)
    D.Dat["nstrokes_beh"] = dfdat["nstrokes_beh"]
    D.Dat["nstrokes_task"] = dfdat["nstrokes_task"]
    for i in range(n_strok_max):
        D.Dat[f"{i}_shape"] = dfdat[f"{i}_shape"]
        D.Dat[f"{i}_loc"] = dfdat[f"{i}_loc"]
        
    Dcopy = sn.Datasetbeh.copy()

    # Remove if aborted
    Dcopy.filterPandas({"aborted":[False], "task_kind":["prims_on_grid"]}, "modify")
    # Filter trials that dont have enough strokes
    Dcopy.Dat = Dcopy.Dat[Dcopy.Dat["nstrokes_beh"]>=n_strok_max].reset_index(drop=True)

    # Print final
    print("Pruned dataset for _dataset_extract_prune_sequence")
    print(D.Dat["nstrokes_beh"].value_counts())
    print(D.Dat["nstrokes_task"].value_counts())
    print(D.Dat["0_shape"].value_counts())
    print(D.Dat["0_loc"].value_counts())
    print(D.Dat["1_shape"].value_counts())
    print(D.Dat["1_loc"].value_counts())

    return D


def _dataset_extract_prune_rulesw(sn, same_beh_only, 
        n_min_trials_in_each_epoch, remove_baseline_epochs, 
        first_stroke_same_only,
        plot_tasks=False):
    """
    Prep beh dataset before extracting snippets.
    Return pruned dataset (copy), to do rule swtiching analy
    matching motor beh, etc.
    """

    if first_stroke_same_only:
        assert same_beh_only==False, "this will throw out cases with same first stroke, but nto all strokes same."
    D = sn.Datasetbeh.copy()

    ##### Clean up tasks
    # 1) only tasks that are done in all epochs (generally, that span all combo of variables).
    # 2) only tasks with completed trials (not failures).
    # 3) only tasks that are correct (especially for probes).

    ##### Plot just tasks with common first stroke

    ##### Plot just for the tasks with common stim, diff behavior
    # Extract it
#     df = SP.DfScalar
#     # save it
#     dfORIG = SP.DfScalar.copy()

    # only "correct" trials (based on actual sequence executed)
    print("############ TAKING ONLY CORRECT TRIALS")
    inds_correct = []
    inds_incorrect = []
    list_correctness = []
    for ind in range(len(D.Dat)):
        dat = D.sequence_extract_beh_and_task(ind)
        taskstroke_inds_beh_order = dat["taskstroke_inds_beh_order"]
        taskstroke_inds_correct_order = dat["taskstroke_inds_correct_order"]

        if taskstroke_inds_beh_order==taskstroke_inds_correct_order:
            # correct
            inds_correct.append(ind)
            list_correctness.append(True)
        else:
            inds_incorrect.append(ind)
            list_correctness.append(False)

    D.Dat["sequence_correct"] = list_correctness
    print("-- Correctness:")
    D.grouping_print_n_samples(["character", "epoch", "sequence_correct"])

    print("correct, incorrect:", len(inds_correct), len(inds_incorrect))
    D.subsetDataframe(inds_correct)
    print("Dataset len:", len(D.Dat))

    print("############ TAKING ONLY NO SUPERVISION TRIALS")
    LIST_NO_SUPERV = ["off|0||0", "off|1|solid|0", "off|1|rank|0"]
    # Only during no-supervision blocks
    print(D.Dat["supervision_stage_concise"].value_counts())
    D.filterPandas({"supervision_stage_concise":LIST_NO_SUPERV}, "modify")
    print("Dataset len:", len(D.Dat))

    # Only characters with same beh across rules.
    # 2) Extract takss done the same in mult epochs
    if first_stroke_same_only:
        # find lsit of tasks that have same first stroke across epochs.
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        list_tasks = D.Dat["character"].unique().tolist()
        list_epoch = D.Dat["epoch"].unique().tolist()
        info = grouping_append_and_return_inner_items(D.Dat, ["character", "epoch"])
        list_tasks_keep = []
        print("n unique first stroke identities for each task:")
        for task in list_tasks:
            list_orders = []
            for ep in list_epoch:
                if (task, ep) in info.keys():
                    idx = info[(task, ep)][0] # take first task in epoch
                    sdict = D.sequence_extract_beh_and_task(idx)
                    order = sdict["taskstroke_inds_correct_order"]
                    list_orders.append(order)
            n_orders_first_sroke = len(list(set([o[0] for o in list_orders])))
            n_epochs_with_data = len(list_orders)
            print("task, n_orders_first_sroke, n_epochs_with_data ... ")
            print(task, n_orders_first_sroke, n_epochs_with_data)
            if n_orders_first_sroke==1 and n_epochs_with_data>1:
                list_tasks_keep.append(task)
        print("These tasks keep (same first stroke): ", list_tasks_keep)
        D.filterPandas({"character":list_tasks_keep}, "modify")
        print("New len of D:", len(D.Dat))

    if same_beh_only:
        print("############ TAKING ONLY CHAR WITH SAME BEH ACROSS TRIALS")
        if True:
            # Already extracted in taskgroups (probes.py)
            print("Using this filter on taskgroup: same_beh, same_beh-P]")
            print("Starting task groups:")
            print("N chars per group:")    
            for grp, inds in D.grouping_get_inner_items("taskgroup", "character").items():
                print(grp, ":", len(inds))    
                
            print("----")
            print("N trials per group:")
            for grp, inds in D.grouping_get_inner_items("taskgroup").items():
                print(grp, ":", len(inds))
            print("---------------------")    
            D.grouping_print_n_samples(["taskgroup", "epoch", "character"])

            D.filterPandas({"taskgroup":["same_beh", "same_beh-P"]}, "modify")
        else:
            # Old method, obsolste, this is done to define taskgroup in probes.py
            from pythonlib.dataset.dataset_preprocess.probes import _generate_map_taskclass
            mapper_taskname_epoch_to_taskclass = _generate_map_taskclass(D)

            list_task = D.Dat["character"].unique().tolist()
            list_epoch = D.Dat["epoch"].unique().tolist()
            if len(list_epoch)<2:
                print("maybe failed to preprocess dataset to reextract epochs?")
                assert False
            print(list_task)
            print(list_epoch)
            dict_task_orders = {}
            for task in list_task:

                list_inds_each_epoch = []
                for epoch in list_epoch:

                    if (task, epoch) not in mapper_taskname_epoch_to_taskclass.keys():
                        # Then this task has at least one epoch for which it doesnet havet rials.
                        # print((task, epoch))
                        # print(mapper_taskname_epoch_to_taskclass)
                        INCLUDE_THIS_TASK = False
                        # assert False, "should first exclude tasks that are not present across all epochs"
                    else:
                        Task = mapper_taskname_epoch_to_taskclass[(task, epoch)]
                        inds_ordered = Task.ml2_objectclass_extract_active_chunk(return_as_inds=True)
                        list_inds_each_epoch.append(tuple(inds_ordered))

                if plot_tasks:
                    ax = Task.plotStrokes(ordinal=True)
                    ax.set_title(f"{task}")

                dict_task_orders[task] = list_inds_each_epoch
            
            print(dict_task_orders)
            # pull out tasks which have same sequence
            tasks_same_sequence = []
            for taskname, list_seqs in dict_task_orders.items():
                if len(list(set(list_seqs)))==1:
                    tasks_same_sequence.append(taskname)
            print("\nTHese tasks foudn to have same seuqence across epochs: ")
            print(tasks_same_sequence)
            D.filterPandas({"character":tasks_same_sequence}, "modify")
        print("Dataset len:", len(D.Dat))

    # Remove epoch = baseline
    if remove_baseline_epochs:
        # D.Dat[~(D.Dat["epoch"].isin(["base", "baseline"]))]
        indskeep = D.Dat[~(D.Dat["epoch"].isin(["base", "baseline"]))].index.tolist()
        D.subsetDataframe(indskeep)

    # Only if have at least N trials per epoch
    print("############ TAKING ONLY CHAR with min n trials in each epoch")
    # print(D.Dat["character"].value_counts())
    # print(D.Dat["epoch"].value_counts())
    D.grouping_print_n_samples(["taskgroup", "epoch", "character"])
    df = D.prune_min_ntrials_across_higher_levels("epoch", "character", n_min=n_min_trials_in_each_epoch)
    D.Dat = df

    # Final length of D.Dat
    print("Final n trials", len(D.Dat))
    print("Dataset conjunctions:")
    D.grouping_print_n_samples(["taskgroup", "epoch", "character"])

    return D

