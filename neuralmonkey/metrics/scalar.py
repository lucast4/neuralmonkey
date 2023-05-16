""" Score metrics, given dataframes with scalar fr values
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.listtools import sort_mixed_type

class MetricsScalar(object):
    """docstring for ClassName"""
    def __init__(self, data, list_var=None, map_var_to_othervars=None,
            map_var_to_levels=None, list_events_uniqnames=None):
        """
        Generates object to compute values for different ways in which scalar fr
        values are modulated by variables
        PARAMS;
        - data, pd dataframe, must have column "fr_scalar",
        and all items in list_var and keys in map_var_to_othervars
        - list_var, list of str, variables of interest for analyses.
        - map_var_to_othervars, for each variables of interest (keys) map it to
        the column in data that is the conjunction of the other variables in 
        list_var.
        - list_events_uniqnames, lsit of str, these events are values in a data["event_aligned"],
        for each column it is scalar avalue for fr aligned to this event
        """

        self.Data = data

        if list_events_uniqnames is None:
            self.ListEventsUniqname = sorted(data["event_aligned"].unique().tolist())
        else:
            self.ListEventsUniqname = list_events_uniqnames 
        self.ListEventsUniqname = [e for e in self.ListEventsUniqname if e in data["event_aligned"].unique()]

        if list_var is None:
            list_var = []
            # assert False, "code it"

        if map_var_to_othervars is None:
            map_var_to_othervars = {}
        for var, othervar in map_var_to_othervars.items():
            if othervar is not None:
                assert othervar in data.columns
        self.MapVarToConjunctionOthers = map_var_to_othervars

        for var in list_var:
            assert var in data.columns
            # assert var in map_var_to_othervars.keys()
        self.ListVar = list_var

        if map_var_to_levels is None:
            map_var_to_levels = {}
            for var in self.ListVar:
                levels = sort_mixed_type(data[var].unique().tolist())
                try:
                    levels = sorted(levels)
                except TypeError as err:
                    pass
                map_var_to_levels[var] = levels
        for var in list_var:
            assert var in map_var_to_levels
        self.MapVarToLevels = map_var_to_levels

        # Check that all frmat have same time bins, then collect it
        fr_times_mat = np.concatenate(self.Data["fr_sm_times"].tolist(), axis=0)
        assert np.all(np.diff(fr_times_mat, axis=0)<0.001), "some rows have different time bins from each other"
        self.TimesFrSm = self.Data.iloc[0]["fr_sm_times"].squeeze() # (ntimes,)

        # Sort so is increasing trialcode.
        from pythonlib.tools.pandastools import sort_rows_by_trialcode
        self.Data = sort_rows_by_trialcode(self.Data)

    def calc_fr_across_levels(self):
        """ Wrapper to cmpute, for each channel:
        for each var, calculate the mean fr across levels for that var. levels are in the 
        same order, defined by self.MapVarToLevels
        RETURNS:
        - output, dict of results
        """

        output = {}
        data = self.Data
        list_events_uniqnames = self.ListEventsUniqname
        map_var_to_othervars=self.MapVarToConjunctionOthers
        map_var_to_levels = self.MapVarToLevels

        # 1) All events combined
        dict_modulation = {}
        dict_modulation_othervar = {}
        for var in self.ListVar:
            list_levels = map_var_to_levels[var]
            res = _calc_fr_across_levels(data, var, list_levels, map_var_to_othervars)
            dict_modulation[var] = res["all_data"]

            othervars_mat = stack_othervals_values(res["othervars_conjunction"]) # (nothervarlevels, nlevels)
            dict_modulation_othervar[var] = othervars_mat
        output["allevents_alldata"] = dict_modulation
        output["allevents_othervar"] = dict_modulation_othervar

        # 2) split by event
        dict_modulation = {}
        dict_modulation_othervar = {}
        for ev in list_events_uniqnames:
            datathis = data[data["event_aligned"]==ev]

            for var in self.ListVar:
                list_levels = map_var_to_levels[var]
                res = _calc_fr_across_levels(datathis, var, list_levels, map_var_to_othervars)

                # 1. for each event, one value for modulation by this var
                dict_modulation[(ev, var)] = res["all_data"]
                othervars_mat = stack_othervals_values(res["othervars_conjunction"]) # (nothervarlevels, nlevels)
                dict_modulation_othervar[(ev, var)] = othervars_mat
        output["splitevents_alldata"] = dict_modulation
        output["splitevents_othervar"] = dict_modulation_othervar
        
        return output

    def calc_r2_smoothed_fr(self, var, COL_FR = "fr_sm_sqrt", 
            n_shuff = 25, plot_results_summary=False):
        """
        PARAMS:
        - do_shuffle, bool, if True, makes copy of dataset, shuffles, then computes. does not
        affect orig dataset
        """   
        
        PLOT_FR = plot_results_summary

        list_r2 = []
        list_events = self.ListEventsUniqname
        list_r2_shuff_mean = []
        dict_r2_shuffles = {}
        dict_event_var_r2 = {}
        dict_event_var_r2_shuff = {}
        dict_event_var_r2_z = {}
        dict_event_var_r2_minshuff = {}

        for event in list_events:
            
            dfthis = self.Data[(self.Data["event_aligned"]==event)]
            levels_var = self.MapVarToLevels[var]

            assert len(dfthis)>0
            # Dataset
            r2, SS, SST = _calc_modulation_by_frsm(dfthis, var, levels_var, COL_FR,  
                plot_fr=PLOT_FR)
            list_r2.append(r2)
            dict_event_var_r2[(event)] = r2
            
            # shuffle
            tmp = []
            dict_r2_shuffles[event] = []
            for i in range(n_shuff):
                r2, _, _ = _calc_modulation_by_frsm(dfthis, var, levels_var, COL_FR, 
                    do_shuffle=True, plot_fr=False)
                tmp.append(r2)
                dict_r2_shuffles[event].append(r2)
            dict_event_var_r2_shuff[(event)] = tmp
            list_r2_shuff_mean.append(np.mean(tmp))
            

        # 3) zscored
        from pythonlib.tools.statstools import zscore
        # convert to z-score relative shuffle.
        list_r2_zscored = []
        for i, ev in enumerate(list_events):
            vals_shuff = dict_r2_shuffles[ev]
            val = list_r2[i]
            val2 = dict_event_var_r2[(ev)]

            r2_z = zscore(val, vals_shuff)
            list_r2_zscored.append(r2_z)
            dict_event_var_r2_z[(ev)] = r2_z
            dict_event_var_r2_minshuff[(ev)] = val - np.mean(vals_shuff)

            if plot_results_summary:
                print(ev, ' -- ', val, r2_z, np.mean(vals_shuff), np.std(vals_shuff))

        if plot_results_summary:
            
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharey=False)

            # 1) Old version
            if False:
                RES = [x for x in RES_ALL_CHANS if x["chan"]==site][0]
                ax = axes.flatten()[0]
                vals = RES["RES"]["modulation_across_events"]["epoch"]
                ax.plot(list_events, vals, "-ok")
                vals = RES["RES"]["modulation_across_events_subgroups"]["epoch"]
                ax.plot(list_events, vals, "-or")

            # 2) New version, using smoothed fr
            ax = axes.flatten()[1]
            ax.plot(list_events, list_r2, "-ok")
            ax.plot(list_events, list_r2_shuff_mean, "--xr", label="shuffled")
            ax.legend()
            for i, ev in enumerate(list_events):
                ax.plot(np.ones(len(dict_r2_shuffles[ev]))*i, dict_r2_shuffles[ev], 'xr', alpha=0.2)
                
            # 3) zscored
            ax = axes.flatten()[2]
            ax.plot(list_events, list_r2_zscored, "-ok")
            ax.set_title('zscored')
            ax.set_ylim(-5, 5)
            # ax.axhline(0)
            ax.axhline(-3)
            ax.axhline(3)

        return dict_event_var_r2, dict_event_var_r2_shuff, dict_event_var_r2_z, dict_event_var_r2_minshuff

    def calc_modulation_frsm_v2(self, var, levels=None, dfthis=None, event=None,
        COL_FR="fr_sm_sqrt", n_shuff = 40, DEBUG=False):
        """ Calculate modulation (like anova) by a variable across these levels.
        std of smoothed fr / mean_scalar. Then subtract mean over shuffles of label. 
        PARAMS;
        - levels, will try across these levels. is ok if dfthis is missing one.
        RETURNS
        - modulation, scalar
        - shuff_moduldations, list scalar 
        - modulation_minshuff, scal
        NOTE: 
        - not the 
        """

        if levels is None:
            levels = self.MapVarToLevels[var]
        if dfthis is None:
            dfthis = self.Data
        if event is not None:
            dfthis = dfthis[dfthis["event"]==event]

        if len(dfthis)==0:
            print(self.Data["event"].unique())
            print(event)
            assert False

        dfthis = dfthis.reset_index(drop=True)
        # 1) Modulation
        modulation = self._calc_modulation_by_frsm_v2(dfthis, var, levels, COL_FR, False)

        # 2) Shuffles
        shuff_moduldations = []
        for i in range(n_shuff):
            tmp = self._calc_modulation_by_frsm_v2(dfthis, var, levels, COL_FR, True)
            shuff_moduldations.append(tmp)

        # 3) Difference
        modulation_minshuff = modulation - np.mean(shuff_moduldations)

        # 4) zscore
        from pythonlib.tools.statstools import zscore
        modulation_zscore = zscore(modulation, shuff_moduldations)

        if DEBUG:
            print("===== var, ' -- ' , modulation, modulation_minshuff, modulation_zscore, np.mean(shuff_moduldations), np.std(shuff_moduldations")
            print(var, ' -- ' , modulation, modulation_minshuff, modulation_zscore, np.mean(shuff_moduldations), np.std(shuff_moduldations))

        return modulation, shuff_moduldations, modulation_minshuff, modulation_zscore

    def _calc_modulation_by_frsm_v2(self, dfthis, var, levels, COL_FR="fr_sm_sqrt",
                                   do_shuffle = False):
        """ Low-level code for calculating modulations. See 
        calc_modulation_frsm_v2
        PARAMS:
        - do_shuffle, bool, then runs on a copy of dfthis with labels for var shuffle.d
        RETURNS:
        - modulation, scalar
        """

        if levels is None:
            levels = self.MapVarToLevels[var]

        if do_shuffle:
            dfthis = _shuffle_dataset(dfthis, var)

        if len(levels)==0 or len(dfthis)==0:
            print("HEREdasfdsaf")
            print(dfthis[var].unique())
            print(var)
            print(do_shuffle)
            assert False

        # 1) get mean fr for each level
        try:
            list_frmean = []
            for lev in levels:
                dfthisthis = dfthis[dfthis[var]==lev]
                if len(dfthisthis)>0:
                    frmean = self.frmat_calc_mean_fr_from_df(dfthisthis) # (1, ntime)
                    list_frmean.append(frmean) 
                else:
                    print("hererere", var, lev)
            frmean_eachlevel = np.concatenate(list_frmean) # (nlev, ntime)
        except Exception as err:
            print("=====")
            print(dfthis)
            print("=====")
            print(var)
            print("=====")
            print(dfthis[var].tolist())
            print("=====")
            print(levels)
            print("=====")
            print(do_shuffle)
            print("=====")
            print(list_frmean)
            print("=====")
            for lev in levels:
                dfthisthis = dfthis[dfthis[var]==lev]
                print(lev, len(dfthisthis))
                print(type(lev))
            raise err

        # 2) compute difference
        # -- TODO: dtw
        frmod = np.mean(np.std(frmean_eachlevel, axis=0)) # (1,), mean over time of STD(1,ntime)
        frmeanscal = np.mean(frmean_eachlevel[:]) # (1,)
        modulation = frmod/frmeanscal # convert to ratio relative mean fr.
        
        return modulation

    def calc_modulation_by(self):
        """ Wrapper to compute, for each channel, how it is mouldated by 
        each variable, in different ways
        RETURNS:
        - output, dict of results.
        """
        
        output = {}
        data = self.Data
        list_events_uniqnames = self.ListEventsUniqname
        map_var_to_othervars=self.MapVarToConjunctionOthers

        # 1) All events combined
        dict_modulation = {}
        dict_modulation_meanofothervar = {}
        for var in self.ListVar:
            res = _calc_modulation_by(data, var, map_var_to_othervars=map_var_to_othervars)
            dict_modulation[var] = res["all_data"]

            if res["othervars_conjunction"] is None:
                # no other vars
                dict_modulation_meanofothervar[var] = None
            else:
                othervars_mat = stack_othervals_values(res["othervars_conjunction"]) # (nothervarlevels, nlevels)
                dict_modulation_meanofothervar[var] = np.mean(othervars_mat, axis=0)
        output["allevents_alldata"] = dict_modulation
        output["allevents_eachothervar_mean"] = dict_modulation_meanofothervar

        # 2) split by event
        dict_modulation = {}
        dict_modulation_meanofothervar = {}
        for ev in list_events_uniqnames:
            datathis = data[data["event_aligned"]==ev]

            for var in self.ListVar: 
                res = _calc_modulation_by(datathis, var, map_var_to_othervars=map_var_to_othervars)
                    
                # 1. for each event, one value for modulation by this var
                dict_modulation[(ev, var)] = res["all_data"]
                
    #             # 2. for each event, and each conjunction of other vars
    #             dict_modulation_eachsubset[(ev, var)] = dict_mod_othervars.values()
                
                # Modulation (#1) is combination of 3a and 3b:
                # 3a. for each event, mean modulation across other vars
                if res["othervars_conjunction"] is not None:
                    othervars_mat = stack_othervals_values(res["othervars_conjunction"]) # (nothervarlevels, nlevels)
                    dict_modulation_meanofothervar[(ev, var)] = np.mean(othervars_mat, axis=0)
                else:
                    dict_modulation_meanofothervar[(ev, var)] = None
                
                # 3b. for each event, consistency across other vars
                if False:
                    dict_modulation_component_cons[(ev, var)] = _calc_consistency()
        output["splitevents_alldata"] = dict_modulation
        output["splitevents_eachothervar_mean"] = dict_modulation_meanofothervar
        
        return output


    def _calc_consistency(data, of_this_var, across_this_var):
        """
        Get consistency of the function represneting modulation by
        variable <of_this_var> across different levels of the variable
        <across_this_var>. i.e., high consistnecy means modulated in similar
        manner across levels of <across_this_var>
        PARAMS;
        - of_this_var, string, column in data
        - across_this_var, string, column in data
        """
        
        assert False, "look thru, copied fro notebok"
        # 1) pivot, so that rows are the mean fr across the var of interest
        from pythonlib.tools.pandastools import pivot_table
        dfout = pivot_table(dfthisthis, index=[across_this_var], columns=[of_this_var], 
                            values=["fr_scalar"], flatten_col_names=True)

        dfout = dfout.drop(across_this_var, axis=1)

        # get similarity across rows.
        pg.pairwise_corr(dfout)
        
        assert False, '# PROBLEM: corr doesnt work for n vals <=2'

    def frmat_calc_mean_fr_from_df(self, dfthis, COL_FR="fr_sm_sqrt"):
        frmat = np.concatenate(dfthis[COL_FR].tolist(), axis=0)
        assert len(frmat)>0
        frmean = _calc_mean_fr(frmat)
        assert frmean.shape[1]>0
        return frmean

    def modulation_extract_vals_byvar(self, ev, results, list_var=None, ver="othervars_mean"):
        """
        Helper to extract values from results, related to modulation by specific
        variables, across vars, for this event
        PARAMS:
        - results, dict of results from self.calc_modulation_by
        - ev, string, the event to extract for. make it "ALL" to get agg avross events
        - list_var, lsit of str, vars, one for each value you want to extract. None to get
        all
        RETURNS:
        - vals, list of values, matching list_var
        """

        if list_var is None:
            list_var = self.ListVar

        ## ALL EVENTS
        if ev =="ALL" and ver=="alldata":
            vals = [results["allevents_alldata"][var] for var in list_var]
        elif ev == "ALL" and ver=="othervars_mean":
            vals = [results["allevents_eachothervar_mean"][var] for var in list_var]
        ## SPECIFIC EVENTS
        elif ver=="alldata":
            vals = [results["splitevents_alldata"][(ev, var)] for var in list_var]
        elif ver=="othervars_mean":
            vals = [results["splitevents_eachothervar_mean"][(ev, var)] for var in list_var]
        else:
            print(ev, ver)
            assert False

        return vals


    def modulation_extract_vals_byevent(self, var, results, 
        list_events_uniqnames=None, ver="othervars_mean"):
        """
        Extract specific results of modulation by this var, across events 
        PARAMS;
        - var, str, variable name
        - list_events_uniqnames, lsit of string, unique events.
        RETURNS;
        - vals, list of modulations, matching list_events_uniqnames
        """

        if list_events_uniqnames is None:
            list_events_uniqnames = self.ListEventsUniqname

        vals = [self.modulation_extract_vals_byvar(ev, results, [var], ver)[0] for ev in list_events_uniqnames]
        return vals

    def modulation_calc_summary(self, WHICH_VER="sm_fr_zscored"):
        """ GOOD summary of modulation for this data
        """
            
        RES = {}
        list_var = self.ListVar

        results = self.calc_modulation_by() 

        # Variance explained, using firing rates
        list_events_uniqnames = self.ListEventsUniqname
        RES["modulation_across_events_usingsmfr"] = {}
        RES["modulation_across_events_usingsmfr_zscored"] = {}
        for var in list_var:
            dict_event_var_r2, dict_event_var_r2_shuff, dict_event_var_r2_z = self.calc_r2_smoothed_fr(var, COL_FR = "fr_sm_sqrt", 
                n_shuff = 25, plot_results_summary=False)
            RES["modulation_across_events_usingsmfr"][var] = [dict_event_var_r2[ev] for ev in list_events_uniqnames]
            RES["modulation_across_events_usingsmfr_zscored"][var] = [dict_event_var_r2_z[ev] for ev in list_events_uniqnames]
            

        # Modulation, plot across events
        # - for a given var, plot it across events
        RES["modulation_across_events"] = {}
        RES["modulation_across_events_subgroups"] = {}
        for var in list_var:
            # - get array of eta2 across events
            vals = self.modulation_extract_vals_byevent(var, results, ver="alldata")
    #         vals = modulation_extract_vals_byevent(results, var, list_events_uniqnames, "alldata")
            RES["modulation_across_events"][var] = vals # save
            
    #         vals = modulation_extract_vals_byevent(results, var, list_events_uniqnames, "othervars_mean")
            vals = self.modulation_extract_vals_byevent(var, results, ver="othervars_mean")
            RES["modulation_across_events_subgroups"][var] = vals # save
        
        
        # Inconsistency across subgroupings
        RES["inconsistency_across_events"] = {}
        for var in list_var:
            # - get array of eta2 across events
            vals_all = np.array(self.modulation_extract_vals_byevent(var, results, ver="alldata"))
            vals_sub = np.array(self.modulation_extract_vals_byevent(var, results, ver= "othervars_mean"))
        
            # v1) Difference
            if vals_sub is not None and vals_all is not None:
                vals_diff = vals_sub - vals_all
            else:
                vals_diff = np.nan
            
            RES["inconsistency_across_events"][var] = vals_diff
            
    #         # v2) quotient
    #         inconsistency = 1 - vals_all/vals_sub
    #         ax2.plot(list_events, inconsistency, '-o', color=pcol, label=var)
        
        # Modulation, all events vs. modulation, each event
        RES["avgmodulation_across_methods"] = {}
        RES["avgmodulation_across_methods_labels"] = {}
        for var in list_var:    
            list_events_this = [["ALL"], ["ALL"], self.ListEventsUniqname, self.ListEventsUniqname]
            list_ver_this = ["alldata", "othervars_mean", "alldata", "othervars_mean"]
            x_labels = []
            y_vals =[]
            for listev, ver in zip(list_events_this, list_ver_this):
                val = np.mean((self.modulation_extract_vals_byevent(var, results, listev, ver)))
                if listev==["ALL"]:
                    label = f"ALLEV-{ver}"
                else:
                    label = f"SPLITEV-{ver}"

                x_labels.append(label)
                y_vals.append(val)
                
            
            RES["avgmodulation_across_methods"][var] = y_vals
            RES["avgmodulation_across_methods_labels"] = x_labels
            
        # b) mean fr across events (simple)
        list_fr = []
        for ev in self.ListEventsUniqname:
            fr = self.Data[self.Data["event_aligned"] == ev]["fr_scalar"].mean()
            list_fr.append(fr)
        RES["avgfr_across_events"] = list_fr
        
        return RES

    ######################## ANOVA
    # def anova_running_wrapper(self, var, vars_others=None, event=None):
    #     """
    #     Wrapper to use self.Data, with specific event
    #     """

    #     dfthis =

    def _anova_running_wrapper_nosplits(self, dfthis, var, vars_others="vars_others",
        TIME_BIN_SIZE=0.2, agg_method="max_sse", do_minus_shuff=True, n_shuff = 12,
        CHECK_SAMPLE_SIZE=False):
        """ Anova, taking the peta for the time bin that is max. 
        Emprically, with bin size of 0.2, n_shuff of 10 is very good for score minus shuffle,
        even for low-fr and low modulation cases. NOTE: but not good for z-score, there is high
        variance, even for n = 20-30. Probably want like n = 50.
        PARAMS:
        - vars_others, either str col name (2-way anova) or None (1-way)
        - CHECK_SAMPLE_SIZE, code for iterating for differnet n_shuff and plotting, so see where
        variance drops. 
        RETURNS:
        - DICT_SCORE, dict[source]=scalar, partial eta 2
        - DICT_MINSHUFF, same but minus shuffle
        - DICT_ZSCORE, same, but minus z-score
        """

        from pythonlib.tools.statstools import zscore
        # 1) Modulation
        res_dict_peta2 = self._anova_running_wrapper_nosplits_inner(dfthis, var, vars_others, TIME_BIN_SIZE,
            agg_method)

        # 2) Shuffles
        res_dict_peta2_shuff = []
        for i in range(n_shuff):
            tmp = self._anova_running_wrapper_nosplits_inner(dfthis, var, vars_others, TIME_BIN_SIZE,
                agg_method, do_shuffle=True)
            res_dict_peta2_shuff.append(tmp)

        # 3) Difference
        list_source = list(res_dict_peta2.keys())
        DICT_MINSHUFF = {}
        DICT_ZSCORE = {}
        DICT_SCORE = res_dict_peta2
        for s in list_source:
            val = res_dict_peta2[s]
            vals_shuff = [x[s] for x in res_dict_peta2_shuff]

            val_minus_shuff = val - np.mean(vals_shuff)
            val_zscore = zscore(val, vals_shuff)
            DICT_MINSHUFF[s] = val_minus_shuff
            DICT_ZSCORE[s] = val_zscore
            # print(val)
            # print(vals_shuff)
            # print(np.mean(vals_shuff))
            # print(np.std(vals_shuff))
            # assert False

        if CHECK_SAMPLE_SIZE:
            assert False, "run this in notebook."
            RES_ZSCORE = []
            RES_MINSHUFF = []
            for _ in range(5):
                for nshuff in range(5, 20, 2):
                    print(nshuff)
                    DICT_SCORE, DICT_MINSHUFF, DICT_ZSCORE = MS._anova_running_wrapper_nosplits(MS.Data, var, n_shuff=nshuff)
                    DICT_ZSCORE["nshuff"] = nshuff
                    DICT_MINSHUFF["nshuff"] = nshuff
                    RES_ZSCORE.append(DICT_ZSCORE)        
                    RES_MINSHUFF.append(DICT_MINSHUFF)
            #         print(DICT_SCORE)
            #         print(DICT_MINSHUFF)
            #         print(DICT_ZSCORE)

            import seaborn as sns
            # dftmp = pd.DataFrame(RES_ZSCORE)
            dftmp = pd.DataFrame(RES_MINSHUFF)
            fig = sns.catplot(data=dftmp, x="nshuff", y="epoch")
            plt.axhline(0)
            sns.catplot(data=dftmp, x="nshuff", y="vars_others")
            plt.axhline(0)
            sns.catplot(data=dftmp, x="nshuff", y="epoch * vars_others")
            plt.axhline(0)

        return DICT_SCORE, DICT_MINSHUFF, DICT_ZSCORE

    def _anova_running_wrapper(self, dfthis, var, vars_others="vars_others",
            n_iter=None, time_bin_size = None, PLOT=False, event=None, 
            PLOT_RESULTS_STANDARD_ERROR=False):
        """ runs _anova_running_wrapper_inner n times and takes the 
        average. This is needed becuase of the random train_test splitting
        dones in there
        PARAMS:
        - vars_others, leave None for onw-wau
        - n_iter, int, n times to run and take average over all results.
        defualt of 20 is based on empirically checking empriclaly the 
        standard error vs. sample size, and elbow is at about 20, with
        stderror about 0.003-0.006, which is fine since effects range from about
        0.01 to 0.1. This was done for var=epoch, othervars=["taskgroup"], for
        10/20/22 Pancho.
        - PLOT_RESULTS_STANDARD_ERROR, bool, if true, then plots, for n_iter from
        1 to n_iter, the standard error of extracted p_eta across those samples, The idea 
        is to find a low ster, this is how many n_iter you want. See notes above.
        RETURNS:
        - df_res, dataframe holding all results.
        """

        assert (n_iter is None) == (time_bin_size is None)
        if n_iter is None:
            # emprically validated that at this time bin size, 15 iters leads to 
            # asymptote of sterr. To check this, set PLOT_RESULTS_STANDARD_ERROR=True
            time_bin_size = 0.2
            n_iter = 10
            # NOTE:
            # if TIME_BIN_SIZE = 0.1, then N_ITER = 20 is good

        if event is not None:
            dfthis = dfthis[dfthis["event"]==event].reset_index(drop=True)

        RES = []
        for it in range(n_iter):
            # print(f"_anova_running_wrapper... iter {it}")
            dict_peta2 = self._anova_running_wrapper_inner(dfthis, var, 
                vars_others=vars_others, TIME_BIN_SIZE=time_bin_size,
                PLOT=PLOT) 
            
            # STORE across iterations
            RES.append(dict_peta2)    

        df_res = pd.DataFrame(RES)

        # Summarize across iterations.

        if PLOT_RESULTS_STANDARD_ERROR:
            list_check = np.arange(2, n_iter, 2)

            out = []

            for i in range(10):
                for n in list_check:

                    # print(n)
                    import random
                    inds = random.sample(range(len(df_res)), n)

                    df_res_sub = df_res.iloc[inds]

                    # compute standard error
                    for source in df_res_sub.columns:

                        ster = np.std(df_res_sub[source])/(len(df_res_sub)**0.5)
                        out.append({
                            "sample_size":n,
                            "peta2_sterr":ster,
                            "source":source})


                #     this = np.mean(df_res_sub, axis=0)

                #     for source in df_res.columns:
                #         out.append({
                #             "sample_size":n,
                #             "peta2":this[source],
                #             "source":source})

            dfout = pd.DataFrame(out)
            import seaborn as sns
            sns.catplot(data=dfout, x="sample_size", y="peta2_sterr", hue="source")        

            print("RESULTS:")
            print(np.mean(df_res, axis=0))
            assert False

        return df_res

    def _anova_running_results_compute_peta(self, dict_results, source):
        """ Help compute partial eta squared for results from     
        _anova_running_wrapper_inner"""
        ss_effect = np.array(dict_results[source])
        ss_resid = np.array(dict_results["Residual"])
        petas = ss_effect/(ss_effect + ss_resid)
        return petas

    def _anova_running_results_plot(self, dict_results):
        """ Helps plot results from _anova_running_wrapper_inner
        """
        
        fig, axes = plt.subplots(2,2)
            
        # raw ss
        ax = axes.flatten()[0]
        for source, vals_ss_effect in dict_results.items():
            if source=="Residual":
                continue
            ax.plot(vals_ss_effect, label=source)
        ax.legend()
        
        
        # raw ss (residuals)
        ax = axes.flatten()[2]
        ax.plot(dict_results["Residual"], label="Residual")
        
        
        ss_all = []
        for source, vals in dict_results.items():
            ss_all.append(vals)
        ss_total = np.sum(np.stack(ss_all), axis=0)
        ax.plot(ss_total, label="Total")
        ax.legend()
        ax.set_ylim(0)
        
        # peta
        ax = axes.flatten()[1]
        for source, vals_ss_effect in dict_results.items():
            if source=="Residual":
                continue
            petas = self._anova_running_results_compute_peta(dict_results, source)
            ax.plot(petas, label=source)
        ax.legend()

    def _anova_running_wrapper_nosplits_inner(self, dfthis, var, vars_others="vars_others", 
            TIME_BIN_SIZE = 0.2, agg_method="max_sse", PLOT=False, do_shuffle = False):
        """
        Wrapper for different methods for summarizing running anova into a single partial-eta squared
        value, wihtout splitting data into train/test.
        PARAMS;
        - var, str
        - vars_others, str, column name --> 2-way anova.
        - agg_method, str, method.
        --- max_sse, take the peta2 at the timebin with max sse, Collect the sse separately for each source.
        but compute peta by normalizing to the mean residual error across the sources.
        RETURNS:
        - res_dict_peta2, dict holding scalar p-eta-sq for each source (var, var_others, var_interaction)
            {'epoch': 0.01647922097755133,
             'vars_others': 0.0509808147337901,
             'epoch * vars_others': 0.029849446813140834}
        """

        assert agg_method=="max_sse", "not yet coded for others"

        if do_shuffle:
            if vars_others is None:
                # One way
                dfthis = _shuffle_dataset(dfthis, var)
            else:
                # two way
                dfthis = _shuffle_dataset_varconj(dfthis, [var, vars_others])

        # Compute running anova to find max indices, using training set
        # print("_anova_running_compute")
        dict_results, times, dict_peta2 = self._anova_running_compute(dfthis, var, 
            vars_others=vars_others, time_bin_size=TIME_BIN_SIZE, return_peta2=True)

        if PLOT:
            self._anova_running_results_plot(dict_results)
            
        # Using training data, find min/max ind. Then extract peta for the test data
        list_source = dict_results.keys()
        dict_ss_effects = {}
        dict_ss_resid = {} # collect all residuals
        # list_residuals = []
        res_dict_peta2 = {}
        for s in list_source:

            if s=="Residual":
                continue

            indmax = np.argmax(dict_results[s]) # maximum ss effect
            peta2_max = dict_peta2[s][indmax]
            
            if PLOT:
                print(f"max ind for {s} = {indmax}")
                print(f"peta2: {peta2_max}")

            res_dict_peta2[s] = peta2_max
            dict_ss_effects[s] = dict_results[s][indmax]
            dict_ss_resid[s] = dict_results["Residual"][indmax]

        # Take the mean residual as the single global residual
        ss_resid_mean = np.mean(list(dict_ss_resid.values()))

        # FINALLY, compute peta for each source
        dict_peta2_commonresidual = {}
        for source, ss_effect in dict_ss_effects.items():
    #         dict_peta2[source] = ss_effect/(ss_effect + ss_resid_max)
            dict_peta2_commonresidual[source] = ss_effect/(ss_effect + ss_resid_mean)

        if PLOT:
            print("----")
            print("EFFECTS:", dict_ss_effects)
            print("RESID:", dict_ss_resid)
            print("PETA2 (using resid for each own tbin):", res_dict_peta2)
            print("PETA2 (using common ss resid):", dict_peta2_commonresidual)

        return dict_peta2_commonresidual


    def _anova_running_wrapper_inner(self, dfthis, var, vars_others="vars_others", 
            TIME_BIN_SIZE = 0.1, PLOT=False, test_size=0.75):
        """ Derived method to compute anova for modulation by var and otherwvars,
        two-way anova:
        - taking tiomepoint of max ss_effect (cross-validated).
        - usings single global ss_residual for all sources (mean).
        RETURNS:
        - dict_peta2, dict[source]=peta2 (partial eta squared)
        eg for two-way:
            {'epoch': 0.011414059046668024,
             'vars_others': 0.027771630853128372,
             'epoch * vars_others': 0.017908765097542172}
        eg for one-way
            {'epoch': -0.02727419836036147}
        """

        # Get splits
        dfthis_train, dfthis_test = self.datamod_split_train_test_stratified(dfthis, var, 
            vars_others=vars_others, test_size=test_size, shuffle=True, DEBUG=False)

        # # Extract data (gets time windows)
        # MS_train = SPall._dataextract_as_metrics_scalar(dfthis_train, var=var)
        # MS_test = SPall._dataextract_as_metrics_scalar(dfthis_test, var=var)
        
        # Compute running anova to find max indices, using training set
        # print("_anova_running_compute")
        dict_results_train, times = self._anova_running_compute(dfthis_train, var, 
            vars_others=vars_others, time_bin_size=TIME_BIN_SIZE) 
        for k, v in dict_results_train.items():
            assert len(v)==len(times), "this needed, as I take index into teims later."
        if PLOT:
            dict_results_test, times = self._anova_running_compute(dfthis_test, var, 
                vars_others=vars_others, time_bin_size=TIME_BIN_SIZE)
            self._anova_running_results_plot(dict_results_train)
            self._anova_running_results_plot(dict_results_test)
            
        # Using training data, find min/max ind. Then extract peta for the test data
        list_source = dict_results_train.keys()
        dict_peta2 = {}
        out = []
        dict_ss_effects = {}
        dict_ss_resid = {} # collect all residuals
        list_residuals = []
        # print("_anova_running_compute_single")
        for s in list_source:

            if s=="Residual":
                continue
            
            # print(dict_results_train)
            # print(s)
            # print(dict_results_train[s])
            # print(len(dict_results_train[s]))
            indmax = np.argmax(dict_results_train[s])
            indmin = np.argmin(dict_results_train[s])
            
            if PLOT:
                print(f"min/max ind for {s} = {indmin}/{indmax}")

            # use this index in the held out data
            # timemin = self.TimesFrSm[indmin]
            # timemax = self.TimesFrSm[indmax]
            timemin = times[indmin]
            timemax = times[indmax]

            res_min, _= self._anova_running_compute(dfthis_test, var, 
                vars_others=vars_others, time_bin_size=TIME_BIN_SIZE,
                single_time_sec=timemin)
            res_max, _= self._anova_running_compute(dfthis_test, var, 
                vars_others=vars_others, time_bin_size=TIME_BIN_SIZE,
                single_time_sec=timemax)

            # res_min = self._anova_timebin_compute(dfthis_test, var, timemin, vars_others=vars_others)
            # res_max = self._anova_timebin_compute(dfthis_test, var, timemax, vars_others=vars_others)

            if False:
                # Stopped this, since this doesnt make sense, e.g.,
                # if moidulation high thorughht (in actuality) this
                # penalizes the score.
                # result is max minus min.
                dict_ss_effects[s] = res_max[s] - res_min[s]
                # dict_ss_effects[s] = res_max[s] - res_min[s]
                # dict_ss_resid[s] = res["Residual"]
                list_residuals.append(res_min["Residual"])
                list_residuals.append(res_max["Residual"])
            else:
                dict_ss_effects[s] = res_max[s]
                list_residuals.append(res_max["Residual"])

        # Take the mean residual as the single global residual
        # ss_resid_mean = np.mean(list(dict_ss_resid.values()))
        ss_resid_mean = np.mean(list_residuals)

        # FINALLY, compute peta for each source
        dict_peta2 = {}
        for source, ss_effect in dict_ss_effects.items():
    #         dict_peta2[source] = ss_effect/(ss_effect + ss_resid_max)
            dict_peta2[source] = ss_effect/(ss_effect + ss_resid_mean)

        if PLOT:
            print("----")
            print("EFFECTS:", dict_ss_effects)
            print("RESID:", dict_ss_resid)
            print("PETA2:", dict_peta2)

        return dict_peta2


    # def _anova_timebin_compute(self, dfthis, var, tget_sec, vars_others="vars_others"):
    #     """ Given a single timebin (sec, rel alignment),
    #     compute anova
    #     RETURNS:
    #     - dict_results, holding source:SS, scalars
    #     """

    #     assert False, "use _anova_running_compute, with flag singe_time_sec"
    #     import pingouin as pg
        
    #     # Extract data
    #     frvec, _, _ = self._dataextract_frvec_single_timebin(dfthis, tget_sec)
    #     var_vals = dfthis[var].tolist()
    #     if vars_others:
    #         vars_others_vals = dfthis[vars_others].tolist()
        
    #     # Get anova
    #     if vars_others:
    #         list_sources = [var, vars_others, f"{var} * {vars_others}", "Residual"]
    #     else:
    #         list_sources = [var, "Within"]

    #     dict_results = {}
    #     # for s in list_sources:
    #     #     dict_results[s] = []

    #     if vars_others:
    #         dftmp = pd.DataFrame({"fr":frvec, var:var_vals, vars_others:vars_others_vals})
    #         BETWEEN = [var, vars_others]
    #     else:
    #         dftmp = pd.DataFrame({"fr":frvec, var:var_vals})
    #         BETWEEN = [var]

    #     aov = pg.anova(data=dftmp, dv="fr", between=BETWEEN, detailed=True, effsize="n2")

    #     # keep results
    #     for s in list_sources:
    #         SS = aov[aov["Source"]==s]["SS"].values[0]
    #         # dict_results[s] = SS
    #         if s=="Within":
    #             # for one-way, it is called Within.
    #             dict_results["Residual"] = SS
    #         else:
    #             dict_results[s] = SS

    #     return dict_results

    def _anova_running_compute(self, dfthis, var, vars_others="vars_others", 
            fr_ver="fr_sm_sqrt", time_bin_size=0.2, single_time_sec=None, return_peta2=False):
        """ Get running two-way anova, var and othervars
        If vars_others is None, then does one-way (var). Otherwise is two-way (var, vars_others)
        PARAMS:
        - single_time_sec, then just gets anova for this vector, that closest to this timepoint.
        RETURNS:
        - dict_results[source] = list of SS_effects.
        - times, array (ntimes,). if single_time_sec, then this is len 1.
        e.g., for single_time_sec not None:
            dict_results = {'epoch': [3.7145323462759294],
              'vars_others': [5.437784095076899],
              'epoch * vars_others': [1.1904845056946296],
              'Residual': [9940.092048615552]}
            times = array([0.3500000000000001])
        """
        import pingouin as pg
            
        if vars_others is not None:
            assert isinstance(vars_others, str), "this must be the the column name, not list of vars"

        # Extract data
        # print("_dataextract_as_frmat")
        frmat, times = self._dataextract_as_frmat(dfthis, fr_ver, time_bin_size=time_bin_size)
        var_vals = dfthis[var].tolist()
        if vars_others is not None:
            vars_others_vals = dfthis[vars_others].tolist()

        # Get running anova.
        ntimes = frmat.shape[1]
        assert ntimes == len(times)
            
        # Initialize results
        dict_results = {}
        dict_peta2 = {}
        if vars_others is not None:
            list_sources = [var, vars_others, f"{var} * {vars_others}", "Residual"]
        else:
            list_sources = [var, "Within"]

        for s in list_sources:
            if s=="Within":
                dict_results["Residual"] = []
                dict_peta2["Residual"] = []
            else:
                dict_results[s] = []
                dict_peta2[s] = []

        if single_time_sec is None:
            list_time_bins = range(ntimes)
        else:
            # get the bin that is closes to this single time
            _frvec, _ind, _time = self._dataextract_frvec_single_timebin_frmat(frmat, times, single_time_sec)
            list_time_bins = [_ind]
            times = [_time]

        for i in list_time_bins:
            
            frvec = frmat[:, i]
            if vars_others:
                dftmp = pd.DataFrame({"fr":frvec, var:var_vals, vars_others:vars_others_vals})
                BETWEEN = [var, vars_others]
            else:
                dftmp = pd.DataFrame({"fr":frvec, var:var_vals})
                BETWEEN = [var]

            # print("pg.anova")
            aov = pg.anova(data=dftmp, dv="fr", between=BETWEEN, detailed=True, effsize="n2")
            
            # keep results
            for s in list_sources:
                # print(s)
                # print("HERE",   aov[aov["Source"]==s]["SS"].values)
                SS = aov[aov["Source"]==s]["SS"].values[0]
                peta2 = aov[aov["Source"]==s]["n2"].values[0]

                if s=="Within":
                    # for one-way, it is called Within.
                    dict_results["Residual"].append(SS)
                    dict_peta2["Residual"].append(peta2)
                else:
                    dict_results[s].append(SS)
                    dict_peta2[s].append(peta2)


        # print(frmat.shape)
        # for source, dat in dict_results.items():
        #     print(source, len(dat))
        # assert False

        for k, v in dict_results.items():
            assert len(v)==len(times), "this needed, as I take index into teims later."

        if return_peta2:
            return dict_results, times, dict_peta2
        else:
            return dict_results, times


    ########################### MODULATION BY TIME (e.g., event-aligned)
    def modulationbytime_calc_this(self, site, list_event=None, n_min_trials=10,
        do_zscore_shuffle=True, nshuff=25, nboot = 80):
        """ For each event, comput the modulation for each site (by temporal structure)
        Modulation by time (i.e., how strongly sm fr is consistenyl modulated in same way
        across iterations of this speciifc event, and each level of this var).
        Auto use same sample size across events, using bootstrap.
        PARAMS:
        - list_event, list of str (00_...), if empty, then gets all events in column "event_aligned"
        - var, str, if not None, then gets each level for this var. 
        (NOTE: gets each combo of event and var_level). if None, then ignores var.
        - n_min_trials, if any slice of data has fewer than this, then throws out.
        - do_zscore_shuffle, bool, if True, then gets zscore rel shuffled trials.
        - nshuff, shuffs, to compute do_zscore_shuffle
        - nboot, num bootstraps. Auto use same sample size across events, using bootstrap.
        RETURNS:
        - res, list of dicts, one for each event x var_level
        """
        
        if list_event is None:
            list_event = self.ListEventsUniqname
        
        res = []
        dict_zscores_per_ndat = {}
        frmat_all = self.dataextract_as_frmat(site, event=None)

        # first, determine the sample size for each event.
        # use the min sample size for all data (taking bootstraps if needed).
        list_n = []
        for event in list_event:
            frmat = self.dataextract_as_frmat(site, event) 
            n = frmat.shape[0]
            if n<n_min_trials:
                continue
            list_n.append(n)
        nmin = min(list_n)
        for event in list_event:
                
            frmat = self.dataextract_as_frmat(site, event) 
            
            # minimum size data
            if frmat.shape[0]<n_min_trials:
                if False:
                    print(f"Skipping {event}, because too few trials ({frmat.shape[0]})")
                continue
            
            frac_diff = frmat.shape[0]/nmin
            if (frac_diff>1.03) or (frac_diff<0.97):
            # if False:
                # Then do bootstrap
                list_r2 = []
                for i in range(nboot):
                    # slice subset of frmat_all
                    inds_rand = np.random.choice(frmat.shape[0], size=(nmin,))
                    frmat_sub = frmat[inds_rand]

                    r2_sub = _calc_modulation_by_frsm_event_aligned_time(frmat_sub)
                    list_r2.append(r2_sub)
                r2_actual = np.mean(list_r2)
            else:
                r2_actual = _calc_modulation_by_frsm_event_aligned_time(frmat)

            ################## Compute shuffles
            if do_zscore_shuffle:
                # ndat = frmat.shape[0]
                ndat = nmin
                if ndat not in dict_zscores_per_ndat.keys():
                    
                    # Then compute mean and std for this many datapts.
                    list_r2 = []
                    for i in range(nshuff):

                        # slice subset of frmat_all
                        inds_rand = np.random.choice(frmat_all.shape[0], size=(ndat,))
                        frmat_sub_shuff = frmat_all[inds_rand]

                        r2_shuff = _calc_modulation_by_frsm_event_aligned_time(frmat_sub_shuff)
                        list_r2.append(r2_shuff)
                    
                    meanthis = np.mean(list_r2)
                    stdthis = np.std(list_r2)
                    
                    dict_zscores_per_ndat[ndat] = (meanthis, stdthis)

                meanthis, stdthis = dict_zscores_per_ndat[ndat]

                # - do zscore
                r2_zscored = (r2_actual - meanthis)/stdthis
                r2_minusmean = r2_actual - meanthis
            else:
                r2_zscored = np.nan
                r2_minusmean = np.nan

            ############# SAVE OUTPUT
            res.append({
                "site":site,
                # "region":region,
                "event":event,
                "r2_time":r2_actual,
                "r2_time_zscored":r2_zscored,
                "r2_time_minusmean":r2_minusmean
            })

        return res

    def datamod_split_train_test_stratified(self, dfthis, var, vars_others="vars_others", 
            test_size=0.5, shuffle=True, DEBUG=False):
        """ Split dataset into train and test, making sure to stratify by levels
        of conjunction of (var, vars_others)
        RETURNS:
        - dfthis_train
        - dfthis_test
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index, append_col_with_index_in_group
        from sklearn.model_selection import train_test_split

        # print(dfthis[vars_others])
        # assert False
        # 1. Append column with level in conjucntion (var, othervar)
        if vars_others is None:
            dfthis["var_othervar_conj"] = dfthis[var]
        else:
            dfthis = append_col_with_grp_index(dfthis, grp=[var, vars_others], new_col_name="var_othervar_conj")

        # 2. for each combo of (var, othervar) take  subset
        dfthis_train, dfthis_test = train_test_split(dfthis, 
            test_size=test_size, shuffle=shuffle, stratify=dfthis["var_othervar_conj"])

        if DEBUG:
            print(dfthis["test"].value_counts())
            print(dfthis_train["test"].value_counts())
            print(dfthis_test["test"].value_counts())
        
        return dfthis_train, dfthis_test

    def _dataextract_as_frmat(self, dfthis, fr_ver="fr_sm", time_bin_size=None):
        """
        Helper to extract frmat across all data, with option to further bin
        (time_bin_size).
        PARAMS:
        - time_bin_size, if sec, then bins starting from left time edge
        RETURNS:
        - frmat, 
        - times
        """

        frmat = np.concatenate(dfthis[fr_ver].tolist(), axis=0)    
        times = self.TimesFrSm

        if time_bin_size:

            MINDUR = time_bin_size/4;
            # MINDUR = 0.05

            binedges = np.arange(times[0], times[-1]+time_bin_size, time_bin_size)
            inds_bin = np.digitize(times, binedges)
            inds_bin_unique = np.sort(np.unique(inds_bin))

            list_t =[]
            list_frvec = []
            for binid in inds_bin_unique:
                indsthis = inds_bin==binid
                
                if sum(indsthis)==0:
                    continue
                    
                timesthis = times[indsthis]
                dur = max(timesthis) - min(timesthis)
                if dur<MINDUR:
                    # print("Skipping bin: ", binid, dur)
                    continue
                frmatthis = frmat[:, indsthis]
                
                t = np.mean(timesthis)
                frvec = np.mean(frmatthis, axis=1)
                
                list_t.append(t)
                list_frvec.append(frvec.T)
                
                
            # concat
            times = np.stack(list_t) # (nbins, )s
            frmat = np.stack(list_frvec, axis=1) # (ndat, nbins)
        
        return frmat, times

    def _dataextract_frvec_single_timebin_frmat(self, frmat, times, tget):
        """ Extract the fr (from smoothed fr) at this specific time bin.
        PARAMS:
        - frmat, array (ndata, ntimes)
        - times, array (ntimes, )
        - tget, scalar, the time which you get the closest bin for.
        RETURNS:
        - frvec, (ndat,) vector of fr at this time slice
        - ind, the index into time bins
        - actual time extracted
        """

        assert frmat.shape[1] == len(times)
        ind = np.argmin(np.abs(times - tget))
        frvec = frmat[:, ind]
        return frvec, ind, times[ind]

    def _dataextract_frvec_single_timebin(self, dfthis, tget, fr_ver="fr_sm_sqrt"):
        """ Extract the fr (from smoothed fr) at this specific time bin.
        PARAMS:
        - tget, time bin to get, in sec relative to alignemetn. will
        find the closest bin.
        RETURNS:
        - frvec, (ndat,) vector of fr at this time slice
        - ind, the index into time bins
        - actual time extracted
        """

        frmat, times = self._dataextract_as_frmat(dfthis, fr_ver)
        return self._dataextract_frvec_single_timebin_frmat(frmat, times, tget)

        # ind = np.argmin(np.abs(self.TimesFrSm - tget))

        # assert frmat.shape[1] == len(self.TimesFrSm)

        # frvec = frmat[:, ind]

        # return frvec, ind, self.TimesFrSm[ind]


    def dataextract_as_frmat(self, chan, event=None, fr_ver="fr_sm"):
        """ 
        Extract frmat from self.DfScalar, stacking all instances of this event, and
        (optionally) only this level for this var.
        PARAMS
        - chan, int
        - event, unique event (00_..) into event_aligned, or tuple, in which case combines events in tuple.
        RETURNS:
        - frmat, (ntrials, ntime)
        - times, (ntime, )
        """
                
        if event is None:
            dfthis = self.Data[(self.Data["chan"]==chan)]
        elif isinstance(event, (list, tuple)):
            for ev in event:
                assert ev in self.Data["event_aligned"].unique()
            dfthis = self.Data[(self.Data["chan"]==chan) & (self.Data["event_aligned"].isin(event))]   
        elif isinstance(event, str):
            dfthis = self.Data[(self.Data["chan"]==chan) & (self.Data["event_aligned"]==event)]   
        else:
            print(event)
            print(self.Data.columns)
            assert False, "what type is this event?"
        
        dfthis = dfthis.reset_index(drop=True)
        return self._dataextract_as_frmat(dfthis, fr_ver) 

    def modulationgood_wrapper_plotexamples(self, site, var, vars_conjuction):
        """ Helper to quickly plot example rasters, sm fr, and modulation scores,
        comparing different modulations cores. 

        """
        
        dfthis = self.dataextract_as_df_conjunction_vars(var, vars_conjuction[:1], site)[0]
        # sp.plotgood_rasters_smfr_each_level_combined(site, var, vars_conjuction[:1]);
        ms = self._dataextract_as_metrics_scalar(dfthis)

        levels = ms.MapVarToLevels[var]
        COL_FR = "fr_sm"
        modulation, shuff_moduldations, modulation_minshuff = calc_modulation_by_frsm_v2(dfthis, var, levels, COL_FR, n_shuff=50)

        vars_others_this = [vars_others[0]]
        self.plotgood_rasters_smfr_each_level_combined(site, var, vars_others_this);

        _, lev_df, _ = self.dataextract_as_df_conjunction_vars(var, vars_others_this, site)

        df[var].value_counts()

        for lev, df in lev_df.items():
            print(lev)
            modu, shuff_modu, modu_min_shuff = calc_modulation_by_frsm_v2(df, var, levels, COL_FR)
            print(modu_min_shuff)

        print(self.modulationgood_compute_wrapper(var, vars_others_this, list_site=[site]))


    def modulationgood_wrapper_twoway(self, var, version, vars_others="vars_others"):
        """ Wrapper to help call different subfunctions for computing modulation 
        by a var and vars_others. 
        PARAMS:
        - return_as_score_zscore_tuple, bool, then returns a dict where each value is 
        a tuple (score, zscore).
        - vars_others, str name of a single column holding conjunction variable.
        RETURNS:
        - eventscores, dict, keys are events and values are modulation scores.
        """


        assert vars_others is not None, "You shoudl use modulationgood_wrapper_ instead"
        if version=="r2smfr_running_maxtime_twoway":
            # running anova, then take peta2 at max timepoint (cross-validated)
            # minus peta2 at min timepoint.
            eventscores = {}
            for event in self.ListEventsUniqname:
                res = self._anova_running_wrapper(self.Data, var=var, 
                    vars_others=vars_others, event=event)

                val = np.mean(res[var]).item()
                val_others = np.mean(res[vars_others]).item()
                val_interaction = np.mean(res[f"{var} * {vars_others}"]).item()

                eventscores[event] = (val, val_others, val_interaction)
        elif version=="r2_maxtime_2way_mshuff":
            """ Max time of r2, without splitting, and subtracting shuffled data."""
            eventscores = {}
            for event in self.ListEventsUniqname:
                dfthis = self.Data[self.Data["event"]==event]
                _, DICT_MINSHUFF, _ = self._anova_running_wrapper_nosplits(dfthis, var, 
                        vars_others=vars_others)
                eventscores[event] = (DICT_MINSHUFF[var], DICT_MINSHUFF[vars_others], DICT_MINSHUFF[f"{var} * {vars_others}"])
        else:
            print(version)
            assert False, "code it"

        return eventscores

    def modulationgood_wrapper_(self, var, version, return_as_score_zscore_tuple=True):
        """ Wrapper to help call different subfunctions for computing modulation 
        by a var
        PARAMS:
        - return_as_score_zscore_tuple, bool, then returns a dict where each value is 
        a tuple (score, zscore).
        RETURNS:
        - eventscores, dict, keys are events and values are modulation scores.
        """

        if return_as_score_zscore_tuple:
            assert version in ["fracmod_smfr_minshuff", "r2smfr_running_maxtime_oneway", "r2_maxtime_1way_mshuff"], "code it!"

        if version=="r2smfr_minshuff":
            # r2 using smoothed fr, minus shuffled
            eventscores = self.calc_r2_smoothed_fr(var)[3]
        elif version=="r2smfr_zscore":
            # r2 using smoothed fr, zscored
            eventscores = self.calc_r2_smoothed_fr(var)[2]
        elif version=="fracmod_smfr_minshuff":
            # std as fraction of mean fr, minus shuffle.
            eventscores = {}
            for event in self.ListEventsUniqname:
                res = self.calc_modulation_frsm_v2(var, event=event) 
                if return_as_score_zscore_tuple:
                    # then get (score, zxcore)
                    eventscores[event] = (res[2], res[3])
                else:
                    eventscores[event] = res[2]
        elif version=="fracmod_smfr_zscore":
            # std as fraction of mean fr, minus shuffle.
            eventscores = {}
            for event in self.ListEventsUniqname:
                eventscores[event] = self.calc_modulation_frsm_v2(var, event=event)[3]
        elif version=="r2scal_minshuff":
            assert False, "code it..."
            # _calc_modulation_by

        elif version=="r2_maxtime_1way_mshuff":
            # running anova, then take max peta2, and substract the mean over shuffles
            eventscores = {}
            for event in self.ListEventsUniqname:
                dfthis = self.Data[self.Data["event"]==event]
                DICT_SCORE, DICT_MINSHUFF, DICT_ZSCORE = self._anova_running_wrapper_nosplits(dfthis, var, 
                        vars_others=None)
                if return_as_score_zscore_tuple:
                    eventscores[event] = (DICT_MINSHUFF[var], DICT_ZSCORE[var])
                else:
                    eventscores[event] = DICT_MINSHUFF[var]

        elif version=="r2smfr_running_maxtime_oneway":
            # running anova, then take peta2 at max timepoint (cross-validated)
            # minus peta2 at min timepoint.
            eventscores = {}
            for event in self.ListEventsUniqname:
                res = self._anova_running_wrapper(self.Data, var=var, 
                    vars_others=None, event=event)
                vals = res[var]
                if return_as_score_zscore_tuple:
                    # then, a bit hacky, compute how many z-scores from data to 0
                    # using the z-score over the data, (not z-score over shuffles, as
                    # I usually do)
                    z = np.mean(vals)/np.std(vals)
                    eventscores[event] = (np.mean(vals).item(), z.item())
                else:
                    eventscores[event] = np.mean(vals).item()
        # elif version=="r2smfr_running_maxtime_twoway":
        #     # running anova, then take peta2 at max timepoint (cross-validated)
        #     # minus peta2 at min timepoint.
        #     eventscores = {}
        #     for event in self.ListEventsUniqname:
        #         res = self._anova_running_wrapper(self.Data, var=var, 
        #             vars_others="vars_others", event=event)

        #         print(res)
        #         assert False
        #         # eventscores[event] = self.calc_modulation_frsm_v2(var, event=event)[3]
        else:
            print(version)
            assert False, "code it"

        if return_as_score_zscore_tuple:
            return eventscores
        else:
            return eventscores

####################### UTILS
def _shuffle_dataset_varconj(df, list_var, maintain_block_temporal_structure=True, 
        shift_level="datapt", DEBUG=False, PRINT=False):
    """ Like _shuffle_dataset, but allowing you maintain the correlation between
    multiple varialbes. e..g, if you are doing two-way anova, want to make sure the sample
    size of conjunction var1xvar2 does not change. TO do this, make dummy variable that 
    is conjunciton, shuffle using that varaibale, then pull out new var1 and var2 from 
    dummy
    PARAMS:
    - list_var, list of str, 
    RETURNS:
    - copy of df, with labels for each var in list_var shuffed
    """
    # make a new temp var
    from pythonlib.tools.pandastools import append_col_with_grp_index, applyFunctionToAllRows, grouping_print_n_samples

    assert len(list_var)==2, "not yet coded for >2"

    # 1) Make a dummy conjunction variable.
    dfthis = append_col_with_grp_index(df, list_var, "dummy", use_strings=False)

    # 2) Shuffle
    dfthis = _shuffle_dataset(dfthis, "dummy", maintain_block_temporal_structure, shift_level, DEBUG)

    # 3) Pull out the var1 and var2
    # resassign var and vars_others
    def F(x):
        return x["dummy"][0]
    dfthis = applyFunctionToAllRows(dfthis, F, list_var[0])

    def F(x):
        return x["dummy"][1]
    dfthis = applyFunctionToAllRows(dfthis, F, list_var[1])

    if PRINT:
        print("=== Original, first 5 inds. SHOULD NOT CHANGE")
        print(df[list_var[0]][:5])
        print(df[list_var[1]][:5])

        print("=== Shuffled, first 5 inds. SHOULD CHANGE")
        print(dfthis[list_var[0]][:5])
        print(dfthis[list_var[1]][:5])

        print("=== Orig/Shuffled, n for each conj. SHOULD NOT CHANGE")
        print("-orig")
        grouping_print_n_samples(df, list_var)
        print("-shuffled")
        grouping_print_n_samples(dfthis, list_var)

    return dfthis

def _shuffle_dataset(df, var, maintain_block_temporal_structure=True, 
        shift_level="datapt", DEBUG=False):
    """ returns a copy of df, with var labels shuffled
    NOTE: confirmed that does not affect df
    PARAMS:
    - maintain_block_temporal_structure, bool, if True, then shuffles by
    circular shifting of trials. THis is better if you didn't randopmly interleave 
    trials. It is a better estimate of variance of shuffles. 
    """
    import random
    from pythonlib.tools.stringtools import decompose_string
    from pythonlib.tools.listtools import extract_novel_unique_items_in_order
    from pythonlib.tools.listtools import list_roll

    levels_orig = df[var].tolist()
    # maintain_block_temporal_structure=False
    if maintain_block_temporal_structure:
        # make sure dataframe is in order of trials.

        def tc_to_tupleints(tc):
            """ 
            tc "2022-1-10" --> tupleints (2022, 1, 10)
            """
            this = decompose_string(tc, "-")
            return [int(x) for x in this]

        trialcodes = df["trialcode"].tolist()
        trialcodes_sorted = sorted(trialcodes, key=lambda x: tc_to_tupleints(x))
        if not trialcodes==trialcodes_sorted:
            print("Trialcodes, -- , trialcodes_sorted")
            for t1, t2 in zip(trialcodes, trialcodes_sorted):
                print(t1, t2)
            assert False, "ok, you need to code this. sort dataframe"
        # return the levels in this order
        # print(trialcodes)
        # print(sorted(trialcodes))
        # this = [(x, y) for x, y in zip(levels_orig, trialcodes)]

    # shuffle a copy
    levels_orig_shuff = [lev for lev in levels_orig]
    if maintain_block_temporal_structure:
        if shift_level=="trial":
            # then shift to not break within-trial correlations
            possible_shifts = extract_novel_unique_items_in_order(trialcodes)[1]
            shift = random.sample(possible_shifts, 1)[0]
        elif shift_level=="datapt":
            # shift at any datpt.
            shift = random.randint(0, len(levels_orig)-1) # 0, 1, 2, ... n-1, possible shifts
        else:
            print(shift_level)
            assert False

        # Do shuffle
        # Dont use this. it converts list of tuples to list of list.
        # levels_orig_shuff = np.roll(levels_orig_shuff, -shift).tolist() # negative, so that works for trial.
        levels_orig_shuff = list_roll(levels_orig_shuff, -shift)

        if DEBUG:
            print("trialcodes, levels(orig):")
            for t1, t2 in zip(trialcodes, levels_orig):
                print(t1, t2)   
            # print(possible_shifts)
            # print(levels_orig)
            print(levels_orig_shuff)
            print(shift)
            assert False
    else:
        # independently shuffle eachj location
        random.shuffle(levels_orig_shuff)

    dfthis_shuff = df.copy(deep=False)
    dfthis_shuff[var] = levels_orig_shuff

    if False:
        # dont need this sanity check
        if type(levels_orig_shuff[0])!=type(levels_orig[0]):
            print("orig: ", levels_orig)
            print("shuffed:", levels_orig_shuff)
            print(type(levels_orig_shuff[0]))
            print(type(levels_orig[0]))
            assert False
    
    return dfthis_shuff
    
def _calc_mean_fr(frmat):
    """ get mean fr (vector) for frmat (ntrials x ntimes)
    across trials 
    PARMAS:
    - frmat, (ntrials, ntime)
    RETURNS:
    - frmean, (1, ntime)
    """

    assert len(frmat.shape)==2
    
    xmean = np.mean(frmat, axis=0, keepdims=True) # (1, ntime)
    if len(xmean.shape)!=2:
        print(frmat)
        print(frmat.shape)
        print(xmean)
        print(xmean.shape)
        assert False
    return xmean

def _calc_residuals(frmat, frmean):
    """ REturn array of squared residules, one for each trial, each
    a scalar, take squared differeence from mean fr (timecourse) then 
    take sum over all times.
    - frmat (ntrials, ntime)
    - frmean (1, ntime)
    RETURNS;
    - resid_scalars, (ntrials, 1)
    """
    resid_scalars = np.mean((frmat - frmean)**2, axis=1) # take mean over time.
    return resid_scalars # (ntrials, )


def _calc_modulation_by_frsm_event_aligned_time(frmat):
    """ Variant of R2, asking ho much activity is consistently modulated (across time)
    in same way across trials. Is same asthinking of time bins as the levels 
    in an anova analysis. Is analagous to doing SNR, i.e,, (peak - trough)/variance_after_subtracg_mean,
    but is more interpretable
    PARAMS:
    - frmat, (ntrials, ntime), smoothed frate
    """

    # mean residual after subtracting mean fr timecourse
    frmean = _calc_mean_fr(frmat) # (1, time)
    resid_sub = _calc_residuals(frmat, frmean).mean() # scalar

    # mean resid without subtracting fr timecourse (i.e., no effect of the event)
    frmean_total = np.mean(frmat, keepdims=True)
    resid_total = _calc_residuals(frmat, frmean_total).mean()

    r2 = 1-(resid_sub/resid_total)

    return r2

def _calc_modulation_by_frsm(dfthis, var, levels, COL_FR = 'fr_sm_sqrt',
                             do_shuffle=False,
                             plot_fr=False, plot_results=False):   
    """ Calculate modulation of smoothed fr by variable var, returning
    single scalar value for each event/channel. Uses the smoothed fr,
    instead for first converting to scalar
    PARAMS:
    - dfthis, a dataframe holding each row as chan x event
    - var, column in dfthis
    - do_shuffle, then shuffles the labels of var before computing. 
    """

    # levels = sorted(dfthis[var].unique().tolist())
    # levels = self.MapVarToLevels[var]
    if do_shuffle:
        dfthis = _shuffle_dataset(dfthis, var)

    # quick plot to ensure is correct
    if plot_fr:
        fig, ax = plt.subplots(figsize=(3, 2))
        for lev in levels:
            list_fr = dfthis[dfthis[var]==lev][COL_FR]
            frmean = np.mean(list_fr).squeeze()
            t = np.arange(len(frmean))
            ax.plot(t, frmean, label=lev)
        #     for fr in list_fr:
        #         ax.plot(fr, label=lev)
        ax.set_ylim(0)
        ax.legend()

    
    def _calc_resid_helper(df):
        """ 
        helper to calc resid using df
        """
        frmat = np.concatenate(df[COL_FR].tolist(), axis=0)
        frmean = _calc_mean_fr(frmat) 
        residuals = _calc_residuals(frmat, frmean)
        return residuals
        
    
    # 1. residuals realtive to global mean.
    residuals_total = _calc_resid_helper(dfthis)

    # 2. get residuals for each level
    residuals_levels = []
    for lev in levels:
        dfthisthis = dfthis[dfthis[var]==lev]
        res = _calc_resid_helper(dfthisthis) 
        residuals_levels.extend(res)
    
    # 3. get each level's mean relative to global mean
    if False:
        # WOrks, but not needed . see below for r2 using SS_LEVEL_MEANS
        frmat = np.concatenate(dfthis[COL_FR].tolist(), axis=0)
        xmean_tot = _calc_mean_fr(frmat)
        list_mean_each = []
        for lev in levels:
            dfthisthis = dfthis[dfthis[var]==lev]
            frmat = np.concatenate(dfthisthis[COL_FR].tolist(), axis=0)
            list_mean_each.append(_calc_mean_fr(frmat))
        frmat = np.concatenate(list_mean_each, axis=0)
        resid_lev = _calc_residuals(frmat, xmean_tot)
    
    # Get summary stats
    SS = np.mean(residuals_levels)
    SST = np.mean(residuals_total)
    r2 = 1 - SS/SST
    if False:
        SS_LEVEL_MEANS = np.mean(resid_lev)
        # NOTE: These are equal!!
        print("equal?:", r2, SS_LEVEL_MEANS/SST)
        # r2 = SS_LEVEL_MEANS/SST # note this is same as r2 above

    if plot_results:
        fig, ax = plt.subplots()
        ax.hist(residuals_total, bins=20, histtype="step")
        ax.hist(residuals_levels, bins=20, histtype="step", label="levels")
        ax.legend()
        ax.set_title(f"r2={r2}, SS={SS}, SST={SST}")
    
    return r2, SS, SST
    
        
def _calc_modulation_by(data, by, response_var = 'fr_scalar', 
    map_var_to_othervars=None, n_min_dat = 8):
    """ [GOOD] Calculatio modulation of response_var ba <by>, given a data (df),
    and does so also for all conjunctions of other variables. 
    PARAMS:
    - by, string name of variabel whose levels modulate the reposnse_var
    - response_var, string, name of variable response
    - n_min_dat, if < this, then returns nan
    RETURNS:
    - output, dict holding modulation computed across different slices of
    data, each value being a scalar.
    """
    from pythonlib.tools.checktools import check_is_categorical

    import pingouin as pg
    output = {}
    
    # 1) all data
    def _calc(datathis):
        if len(datathis)<n_min_dat:
            return np.nan
        else:
            datcheck = datathis[by].tolist()[0]
            if check_is_categorical(datcheck):
                # Then do anova
                # print("Categorical variable: ", by)
                aov = pg.anova(data=datathis, dv=response_var, between=by, detailed=True)
                eta2_all = aov[aov["Source"]==by]["np2"].item()
                if "np2" not in aov[aov["Source"]==by].keys():
                    print(aov[aov["Source"]==by])
                    print(aov)
                    print(by)
                    print(datathis)
                    assert False            
                return eta2_all
            else:
                # print("Numerical variable: ", by)
                res = pg.linear_regression(X=datathis[by], y=datathis[response_var])
                if len(res["r2"])<2:
                    print("--- N=", len(datathis[by]), "R2", res["r2"])
                    assert False
                r2 = res["r2"][1].item() # index 0 is intercept.
                return r2

    # Decide if use regression or anova depending on the type of the data
    output["all_data"] = _calc(data)
    # datcheck = data[by].tolist()[0]
    # if check_is_categorical(datcheck):
    #     # Then do anova
    #     print("Categorical variable: ", by)
    #     aov = pg.anova(data=data, dv=response_var, between=by, detailed=True)
    #     eta2_all = aov[aov["Source"]==by]["np2"].item()
    #     output["all_data"] = eta2_all
    # else:
    #     print("Numerical variable: ", by)
    #     res = pg.linear_regression(X=data[by], y=data[response_var])
    #     r2 = res["r2"][1].item() # index 0 is intercept.
    #     output["all_data"] = r2
            
    # 2) for each conjunction of other vars
    if map_var_to_othervars is None:
        dict_mod_othervars = None
        levels = None
    else:
        colname_othervars = map_var_to_othervars[by]
        if colname_othervars is not None:
            # levels = sorted(data[colname_othervars].unique().tolist())
            levels = self.MapVarToLevels[colname_othervars]
            dict_mod_othervars = {}
            for lev in levels:
                # print("level: ", lev, "colname_othervars: ", colname_othervars)
                datathis = data[data[colname_othervars]==lev]
                dict_mod_othervars[lev] = _calc(datathis)
                # aov = pg.anova(data=datathis, dv=response_var, between=by, detailed=True)
                # eta2 = aov[aov["Source"]==by]["np2"].item()
                # dict_mod_othervars[lev] = eta2
        else:
            dict_mod_othervars = None
            levels = None
    output["othervars_conjunction"] = dict_mod_othervars
    output["othervars_conjunction_levels"] = levels

    return output

def _calc_fr_across_levels(data, var, list_levels, map_var_to_othervars=None,
    response_var = "fr_scalar"):
    """ Calculatio mean fr across levels for this var. modulation of response_var ba <by>,
    PARAMS:
    - data, df, usually for a single channel and event.
    - var, string, variable whose levels to compute fr for
    - list_levels, list of str, levels in order desired. asks you to  pass it in, to ensure that
    the resulting order is correct.
    - map_var_to_othervars, to get result for each conjuiction of other vars.
    RETURNS:
    - output, dict holding results, each value being a list of mean frs.
    """
    
    output = {}
    
    def _calc_means(datathis):
        """ Returns list_means matching list_levels. if a lev doesnt have data,
        that mean is np.nan"""
        if False:
            # v1: faster with large dataset
            # took 53ms vs 271ms (below) for agg version (entire dataset, n~1M)
            from pythonlib.tools.pandastools import aggregGeneral
            dfagg = aggregGeneral(datathis, [var], values=response_var)
            for lev in list_levels:
                print(dfagg[dfagg[var]==lev][response_var].item())
            assert False, "get list_means"
        else:
            # faster with small dataset: took 4ms vs. 26 for smaller datsaet (n~2000)
            list_means = []
            for lev in list_levels:
                if np.sum(datathis[var]==lev)==0:
                    # Then doesnt exist
                    list_means.append(np.nan)
                else:
                    # assert np.sum(datathis[var]==lev)>0, f"doesnt exist... {var}, {lev}"
                    m = np.mean(datathis[datathis[var]==lev][response_var])
                    list_means.append(m)
        return list_means
    
    
    # 1) all data
    output["all_data"] = _calc_means(data)
    
    # 2) Each subset, specific conjuiction of other data
    if map_var_to_othervars is None:
        list_means_others = None
        levels_others = None
    else:
        colname_othervars = map_var_to_othervars[var]
        # levels_others = sorted(data[colname_othervars].unique().tolist())
        levels_others = self.MapVarToLevels[colname_othervars]
        list_means_others = {}
        for lev in levels_others:
            datathis = data[data[colname_othervars]==lev]
            list_means = _calc_means(datathis)
            list_means_others[lev] = list_means
    output["othervars_conjunction"] = list_means_others
    output["othervars_conjunction_levels"] = levels_others

    return output




def stack_othervals_values(othervals):
    """ 
    PARAMS;
    - othervals, dict where keys are levels for othervals, and
    values are list of int, or scalars
    RETURNS:
    - np array, stacking the values, with dimensions 
    if inner items are lists : (n to stack, length of inner lists.)
    if inner items are scalars: (n tostack, )
    E.G.:
    {'(0, 0)': [7.740192683973832,
      7.661018549704632,
      7.565683192350642,
      7.942574397711931,
      7.568079304750462,
      7.76416359098134,
      7.651544842664896],
     '(0, 1)': [7.740192683973832,
      7.661018549704632,
      7.565683192350642,
      7.942574397711931,
      7.568079304750462,
      7.76416359098134,
      7.651544842664896]}
    ---> array of shape (2, 7)

    {'(0, 0)': 7.740192683973832,
     '(0, 1)': 7.740192683973832}
    ---> array of shape (2,)
    """

    tmp = list(othervals.values()) # list of lists
    this = np.stack(tmp, axis=0)
    assert this.shape[0]==len(tmp)
    return this