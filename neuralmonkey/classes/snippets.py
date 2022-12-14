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
        prune_feature_levels_min_n_trials=None):
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

        """

        # 1) Which behavioral data to use
        if which_level=="stroke":
            # Each datapt matches a single stroke
            DS = datasetstrokes_extract(SN.Datasetbeh, strokes_only_keep_single,
                tasks_only_keep_these, prune_feature_levels_min_n_trials, list_features_extraction)
        elif which_level=="trial":
            # Each datapt is a single trial
            # no need to extract antyhing, use sn.Datasetbeh
            trials = SN.get_trials_list(True, True)
            pass
        else:
            print(which_level)
            assert False


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

        # 2) Extract snippets
        # [GOOD] Combine those options to extract each relevant window
        # list_events = ["fixcue", "samp", "samp", "first_raise", "STROKE"]
        # list_events = ["fixcue", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        # list_pre_dur = [-0.5, -0.3, 0.1, -0.55, -0.55, 0.05]
        # list_post_dur = [0., 0., 0.65, -0.05, 0.05, 0.55]

        ### Extract snippets.
        ListPA = []
        fig, ax = plt.subplots()
        list_events_uniqnames = []
        map_var_to_othervars_list = []
        for i, (event, pre_dur, post_dur) in enumerate(zip(list_events, list_pre_dur, list_post_dur)):
            
            # 1) Extract single pa
            if which_level=="stroke":
                if event == "STROKE":
                    align_to_stroke=True
                    align_to_alternative = []
                else:
                    align_to_stroke=False
                    align_to_alternative = [event]
                    
                listpa = SN.popanal_generate_alldata_bystroke(DS, sites_keep, align_to_stroke=align_to_stroke, 
                                                              align_to_alternative=align_to_alternative, 
                                                              pre_dur=pre_dur, post_dur=post_dur,
                                                              use_combined_region=False)
            elif which_level=="trial":
                listpa = SN.popanal_generate_alldata(trials, sites_keep,
                    events = [event],
                    pre_dur=pre_dur, post_dur=post_dur, 
                    columns_to_input = list_features_extraction,
                    use_combined_region = False)
            assert len(listpa)==1
            pa = listpa[0]


            # 2) Get conjuctions of features
            map_var_to_othervars = pa.labels_features_input_conjunction_other_vars(dim="trials", 
                list_var = list_features_get_conjunction)

            ListPA.append(pa)
            
            # plot
            ax.plot([pre_dur, post_dur], [i, i], "o-", label=event)
            
            # give event a unique name
            if False:
                # version 1, with time
                event_unique_name = f"{event}|{pre_dur}|{post_dur}"
            else:
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


        ### SAVE VARIABLES
        self.ListPA = ListPA
        self.SN = SN
        self.Sites = sites_keep
        if which_level=="stroke":
            self.DS = DS
            self.Trials = None
        elif which_level=="trial":
            self.DS = None
            self.Trials = trials
        else:
            assert False
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
        self.listpa_convert_to_scalars()
        self.pascal_convert_to_dataframe(fr_which_version="sqrt")

        # Get useful variables
        self._preprocess_map_features_to_levels()

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
        list_pa_scal = []
        list_labels = []
        for i in range(len(PA_scal_all.Chans)):
            pa_scal = PA_scal_all.slice_by_dim_indices_wrapper("chans", [i])
            chan = PA_scal_all.Chans[i]
            list_pa_scal.append(pa_scal)
            list_labels.append(chan)
        PA_scal_all= concatenate_popanals(list_pa_scal, dim="trials", 
                                        map_idxpa_to_value=list_labels, 
                                        map_idxpa_to_value_colname="chan",
                                        assert_otherdims_have_same_values=False)

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


    ################ MODULATION BY VARIABLES
    def modulation_compute_each_chan(self, DEBUG=False, 
        bregion_add_num_prefix=True, 
        bregion_combine=False):
        """ Compute modulation by variables for each channel
        RETURNS:
        - RES_ALL_CHANS, list of dicts
        """
        from neuralmonkey.metrics.scalar import MetricsScalar

        RES_ALL_CHANS = []
        list_chans = self.DfScalar["chan"].unique().tolist()

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
            print(chan)
            info = self.SN.sitegetter_thissite_info(chan)

            dfthis = self.DfScalar[self.DfScalar["chan"]==chan]
            
            # Input to Metrics
            # (use this, instead of auto, to ensure common values across all chans)
            list_var = self.Params["list_features_get_conjunction"]
            list_events_uniqnames = self.Params["list_events_uniqnames"]
            map_var_to_othervars = self.Params["map_var_to_othervars"]
            map_var_to_levels = self.Params["map_var_to_levels"]
            Mscal = MetricsScalar(dfthis, list_var, map_var_to_othervars, 
                map_var_to_levels, 
                list_events_uniqnames)
            
            # Compute, modulation across vars
            RES = Mscal.modulation_calc_summary()

            # Compute, fr across levels, for each var
            RES_FR = Mscal.calc_fr_across_levels()

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
            bregion = RES_ALL["bregion"]
            RES = RES_ALL["RES"]
            RES_FR = RES_ALL["RES_FR"]

            for var in list_var:   

                ########################################
                for val_kind in ["modulation_across_events", 
                    "modulation_across_events_subgroups", 
                    "inconsistency_across_events"]:

                    y = RES[val_kind][var]
                    for ev, yscal in zip(list_events_uniqnames, y):
                        dat_across_var_events.append({
                            "event":ev,
                            "val":yscal,
                            "val_kind":val_kind,
                            "var":var,
                            "chan":chan,
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
                    "bregion":bregion
                })

            ######################## MEAN FR ACROSS LEVELS
            # mod across events
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

    def _modulation_find_first_postsamp_event(self):
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
        print(list_events)
        print(list_pre_dur)
        print(list_post_dur)
        assert False, "did not find..."
 
    def modulation_plot_heatmaps(self, OUT, savedir="/tmp"):
        """ Plot heatmaps, bregion vs. event, and also
        overlay onto schematic of brain, across time(events)
        """
        from pythonlib.tools.pandastools import convert_to_2d_dataframe, aggregGeneral

        df = OUT["dfdat_var_events"]
        val_kind = "modulation_across_events_subgroups"
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
        map_bregion_to_location["06_SMA_p"] = [-.1, 0.2]
        map_bregion_to_location["07_SMA_a"] = [1.4, 0.3]
        map_bregion_to_location["08_dlPFC_p"] = [7.2, 2.8]
        map_bregion_to_location["09_dlPFC_a"] = [9, 3]
        map_bregion_to_location["10_vlPFC_p"] = [5.8, 5]
        map_bregion_to_location["11_vlPFC_a"] = [8.5, 4]
        map_bregion_to_location["12_preSMA_p"] = [3.2, 0.4]
        map_bregion_to_location["13_preSMA_a"] = [4.5, 0.6]
        map_bregion_to_location["14_FP_p"] = [11, 3.9]
        map_bregion_to_location["15_FP_a"] = [12.5, 4.3]
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


    def modulation_plot_all(self, OUT, savedir="/tmp"):
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


        # Compare across events
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(["modulation_across_events", "modulation_across_events_subgroups"])]
        fig = sns.catplot(data=dfthis, x="event", y="val", hue="val_kind",
                          col="bregion", row="var", kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/1_lines_modulation_kinds.pdf")


        # Comparing brain regions
        for val_kind in ["modulation_across_events", "modulation_across_events_subgroups"]:
            dfthis = dfdat_var_events[dfdat_var_events["val_kind"]==val_kind]
            fig = sns.catplot(data=dfthis, x="bregion", y="val", col="event", 
                        row="var", kind="bar", aspect=2, height=4);
            rotateLabel(fig)
            fig.savefig(f"{savedir}/2_bars_feature_vs_event-{val_kind}.pdf")      

        # Comparing brain regions
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(["modulation_across_events", "modulation_across_events_subgroups"])]
        fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind",
                    row="event", kind="bar", hue="var", aspect=2, height=4);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/2_bars_feature_vs_valkind.pdf")

        # dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(["modulation_across_events", "modulation_across_events_subgroups"])]
        # fig = sns.catplot(data=dfthis, x="bregion", y="val", col="event", 
        #             row="var", kind="bar", hue="val_kind", aspect=2, height=4);
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/test2_lowfr.pdf")


        # Compare across events
        dfthis = dfdat_var_events
        fig = sns.catplot(data=dfthis, x="event", y="val", hue="var",
                          col="bregion", row="val_kind", kind="point", aspect=1, height=3);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/3_lines_valkinds.pdf")

        for val_kind in ["modulation_across_events", "modulation_across_events_subgroups"]:
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
        dfthis = dfdat_var_events[dfdat_var_events["val_kind"].isin(["modulation_across_events", "modulation_across_events_subgroups"])]
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
                        raise err


    ########### UTILS   a
    def save(self, sdir, fname="Snippets", add_tstamp=True, exclude_sn=True):
        """ Saves self in directory sdir
        as pkl files
        """
        import pickle as pkl
        if exclude_sn:
            assert False, "not coded"
        if add_tstamp:
            from pythonlib.tools.expttools import makeTimeStamp
            ts = makeTimeStamp()
            fname = f"{sdir}/{fname}-{ts}.pkl"
        else:
            fname = f"{sdir}/{fname}.pkl"

        with open(fname, "wb") as f:
            pkl.dump(self, f)

        print(f"Saved self to {fname}")

    def copy(self, minimal=True):
        """ make a copy, pruning variables that are not needed
        PARAMS:
        - minimal, bool, if True, then copies only the PA objects
        the rest uses reference
        """

        assert False, "have to allow initializing without passing in var"
        vars_to_copy = ["ListPA", "PAscalar"]
        # Params
        # Sites
        # Trials




def datasetstrokes_extract(D, strokes_only_keep_single=False, tasks_only_keep_these=None, 
    prune_feature_levels_min_n_trials=None, list_features=None):
    """ Helper to extract dataset strokes
    """

    # 1. Extract all strokes, as bag of strokes.
    from pythonlib.dataset.dataset_strokes import DatStrokes
    DS = DatStrokes(D)

    if strokes_only_keep_single:
        DS.clean_data(["remove_if_multiple_behstrokes_per_taskstroke"])
    
    if tasks_only_keep_these is not None:
        assert isinstance(tasks_only_keep_these, list)
        # DS.Dat = DS.Dat[DS.Dat["task_kind"].isin(["prims_single", "prims_on_grid"])].reset_index(drop=True)
        DS.Dat = DS.Dat[DS.Dat["task_kind"].isin(tasks_only_keep_these)].reset_index(drop=True)

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
