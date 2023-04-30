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
            print(dfthis[var].unique())
            print(var)
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

    def dataextract_as_frmat(self, chan, event=None, fr_ver="fr_sm"):
        """ 
        Extract frmat from self.DfScalar, stacking all instances of this event, and
        (optionally) only this level for this var.
        PARAMS
        - chan, int
        - event, unique event (00_..) into event_aligned, or tuple, in which case combines events in tuple.
        - var, var_level, either both None (ignore var), or string and value.
        RETURNS:
        - frmat, (ntrials, ntime)
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
            
        frmat = np.concatenate(dfthis[fr_ver].tolist(), axis=0)    
        
        return frmat 

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



    def modulationgood_wrapper_(self, var, version, dfthis=None, return_as_score_zscore_tuple=True):
        """ Wrapper to help call different subfunctions for computing modulation 
        by a var
        PARAMS:
        - return_as_score_zscore_tuple, bool, then returns a dict where each value is 
        a tuple (score, zscore).
        RETURNS:
        - eventscores, dict, keys are events and values are modulation scores.
        """

        if return_as_score_zscore_tuple:
            assert version in ["fracmod_smfr_minshuff"], "code it!"

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
        else:
            print(version)
            assert False, "code it"

        if return_as_score_zscore_tuple:
            return eventscores
        else:
            return eventscores

####################### UTILS
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
        # for t1, t2 in zip(trialcodes, trialcodes_sorted):
        #     print(t1, t2)
        assert trialcodes==trialcodes_sorted, "ok, you need to code this. sort dataframe"
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