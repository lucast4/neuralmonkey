""" Score metrics, given dataframes with scalar fr values
"""
import numpy as np
import pandas as pd

class MetricsScalar(object):
    """docstring for ClassName"""
    def __init__(self, data, list_var, map_var_to_othervars,
            list_events_uniqnames):
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
        self.ListVar = list_var
        self.MapVarToConjunctionOthers = map_var_to_othervars
        self.ListEventsUniqname = list_events_uniqnames

        for var in list_var:
            assert var in data.columns
            assert var in map_var_to_othervars.keys()

        for var, othervar in map_var_to_othervars.items():
            assert othervar in data.columns

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

        # 1) all data
        dict_modulation = {}
        dict_modulation_meanofothervar = {}
        for var in self.ListVar:
            res = _calc_modulation_by(data, var, map_var_to_othervars=map_var_to_othervars)
            dict_modulation[var] = res["all_data"]
            dict_modulation_meanofothervar[var] = np.mean(list(res["othervars_conjunction"].values()))
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
                dict_modulation_meanofothervar[(ev, var)] = np.mean(list(res["othervars_conjunction"].values()))
                
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

    def modulation_calc_summary(self):
        """ GOOD summary of modulation for this data
        """
            
        RES = {}
        list_var = self.ListVar

        results = self.calc_modulation_by()


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
            vals_diff = vals_sub - vals_all
            
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


####################### UTILS
def _calc_modulation_by(data, by, response_var = 'fr_scalar', 
    map_var_to_othervars=None):
    """ Calculatio modulation of response_var ba <by>,
    PARAMS:
    - by, string name of variabel whose levels modulate the reposnse_var
    - response_var, string, name of variable response
    RETURNS:
    - output, dict holding modulation computed across different slices of
    data.
    """
    
    import pingouin as pg
    output = {}
    
    # 1) all data
    aov = pg.anova(data=data, dv=response_var, between=by, detailed=True)
    eta2_all = aov[aov["Source"]==by]["np2"].item()
    output["all_data"] = eta2_all
    
    # 2) for each conjunction of other vars
    if map_var_to_othervars is None:
        dict_mod_othervars = None
        levels = None
    else:
        colname_othervars = map_var_to_othervars[by]
        levels = data[colname_othervars].unique().tolist()
        dict_mod_othervars = {}
        for lev in levels:
            datathis = data[data[colname_othervars]==lev]
            aov = pg.anova(data=datathis, dv=response_var, between=by, detailed=True)
            eta2 = aov[aov["Source"]==by]["np2"].item()
            dict_mod_othervars[lev] = eta2
    output["othervars_conjunction"] = dict_mod_othervars
    output["othervars_conjunction_levels"] = levels

    return output
