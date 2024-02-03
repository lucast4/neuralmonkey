"""
FOr loading and splitting/concatting previously extract PA datasets.
NOWADAYS not used much, since I am not sainvg PA, but isntad goings taight from SP --> Analyses... (saving dsiak sapc).
"""


from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import glob
import pickle
import pandas as pd

# (animal, date, question) --> DFallPA

def dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows,
                                    which_level = "trial",
                                    events_keep = None,
                                    combine_into_larger_areas = True, exclude_bad_areas=True,
                                    bin_by_time_dur = None, bin_by_time_slide = None,
                                    slice_agg_slices = None, slice_agg_vars_to_split=None, slice_agg_concat_dim=None,
                                    LOAD_FROM_RSA_ANALY=False, rsa_ver_dist="euclidian_unbiased",
                                    rsa_subtr=None, rsa_agg = True,
                                    SPIKES_VERSION="tdt",
                                    HACK_RENAME_SHAPES = True):

    """ [GOOD] Hihg level to extrqact
    DFallpa, with all preprocessing steps built in, must have already extgracted Snippets.
    By default this gets separate pa for each (event, bregion), but has many methods for
    slicing and aggregating across multiple PAs.
    """
    from neuralmonkey.scripts.analy_dpca_script_quick import preprocess_pa_to_frtensor
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    from neuralmonkey.classes.session import load_mult_session_helper
    import os
    import pandas as pd
    from neuralmonkey.analyses.state_space_good import snippets_extract_popanals_split_bregion_twind
    from neuralmonkey.analyses.rsa import rsagood_questions_dict

    if LOAD_FROM_RSA_ANALY:
        # Saved in analy_rsa_script.py
        from neuralmonkey.analyses.rsa import rsagood_pa_vs_theor_wrapper_loadresults, rsagood_pa_vs_theor_single
        version_distance = rsa_ver_dist
        subtract_mean_each_level_of_var = rsa_subtr
        DO_AGG_TRIALS = rsa_agg
        DFallpa = rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question,
                                                          version_distance, DO_AGG_TRIALS,
                                                          subtract_mean_each_level_of_var)[0]
    else:
        # Generate it from saved Snippets

        ############### PARAMS
        # animal = "Diego"
        # date = 230615
        # exclude_bad_areas = True
        # SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
        # bin_by_time_dur = 0.05
        # bin_by_time_slide = 0.025

        ## Load Snippets
        MS = load_mult_session_helper(date, animal, spikes_version=SPIKES_VERSION)
        SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level,
            DEBUG=False)

        # Load a question
        DictParamsEachQuestion = rsagood_questions_dict(animal, date)
        q_params = DictParamsEachQuestion[question]

        # Clean up SP and extract features
        D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
            ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"])

        # Keep only specific events - to make the following faster.
        if events_keep is None:
            events_keep = q_params["events_keep"]

        ## Extract all popanals
        DFallpa = snippets_extract_popanals_split_bregion_twind(SP, list_time_windows,
                                                        list_features_extraction,
                                                        HACK_RENAME_SHAPES=HACK_RENAME_SHAPES,
                                                        combine_into_larger_areas=combine_into_larger_areas,
                                                        events_keep=events_keep,
                                                        exclude_bad_areas=exclude_bad_areas)

        # Bin times if needed
        if bin_by_time_dur is not None:
            list_pa = []
            for pa in DFallpa["pa"].tolist():
                list_pa.append(pa.agg_by_time_windows_binned(bin_by_time_dur, bin_by_time_slide))
            DFallpa["pa"] = list_pa

        # Aggregate PA if needed
        from neuralmonkey.classes.population_mult import dfpa_slice_specific_windows, dfpa_group_and_split
        if slice_agg_slices is not None:
            # 1) slice
            print(" *** Before dfpa_slice_specific_windows")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("slice_agg_slices:", slice_agg_slices)
            DFallpa = dfpa_slice_specific_windows(DFallpa, slice_agg_slices)

            # 2) agg (one pa per bregion)
            print(" *** Before dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print(slice_agg_vars_to_split)
            DFallpa = dfpa_group_and_split(DFallpa, vars_to_split=slice_agg_vars_to_split, concat_dim=slice_agg_concat_dim)

            print(" *** After dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("Event, within pa:")

            for pa in DFallpa["pa"].tolist():
                print(pa.Xlabels["trials"]["event"].value_counts())
                print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
                assert isinstance(pa.Xlabels["trials"]["wl_ev_tw"].values[0], str)

    return DFallpa


def load_dataset_mult_wl(animal, date, list_which_level):

    list_out = []
    list_params =[]
    for wl in list_which_level:
        out, params = load_dataset_single(animal, date, wl, return_as_df=True)
        list_out.append(out)
        list_params.append(params)

    DFallpa = pd.concat(list_out).reset_index(drop=True)

    return DFallpa, list_params


def load_dataset_single(animal, date, which_level):
    """ Load a single dataset, which is a dict of popanals, each keyed by
    (event, bregion, twind). WIll first look for PA created from SP concateed
    across mult sessions. if that doesnt eixst, then looks for single session.
    (In progrsss) And concatenates.
    RETURNS:
        - DictEvBrTw_to_PA, dict (which_level, event, bregion, twind): pa.
    """

    # First, look for data made from SP using multiple sessions. (concated SP)
    savedir = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/mult_session"
    path = f"{savedir}/{animal}-{date}-{which_level}-*.pkl"
    files = glob.glob(path)
    print("Found mult session: ", files)

    if len(files)==1:
        mult_or_single = "mult"
    elif len(files)==0:
        # Then look for singles
        savedir = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/single_session"
        path = f"{savedir}/{animal}-{date}-{which_level}-*.pkl"
        files = glob.glob(path)

        print("Found single session: ", files)

        if len(files)==0:
            print()
            assert False, "doesnt exist.."

        mult_or_single = "single"
    else:
        assert False

    ##############
    if len(files)>1:
        print(files)
        assert False, "not yet coded.. should extract session num from filename, then before concat should append sess num as column in all pa.Xlabels[trials]"

        # # Load and concat all files
        # for fi in files:
        #     with open(fi, "rb") as f:
        #         DictEvBrTw_to_PA = pickle.load(f)
    elif len(files)==1:
        # Load and concat all files
        fi = files[0]
        with open(fi, "rb") as f:
            DFallpa = pickle.load(f)
    else:
        assert False

    # # Convert each key from (event, bregion, twind) to (wl, event, bregion, twind)
    # dicttmp = {}
    # for k, v in DictEvBrTw_to_PA.items():
    #     k = tuple([which_level] + list(k))
    #     dicttmp[k] = v
    # DictEvBrTw_to_PA = dicttmp

    # # condition, for entry to dataframe
    # if return_as_df:
    #     tmp = []
    #     for k, v in DictEvBrTw_to_PA.items():
    #
    #         # Make sure pa itself is keeping track of the outer varibles,
    #         # for sanity checks once you start splitting and grouping.
    #         v.Xlabels["trials"]["which_level"] = k[0]
    #         v.Xlabels["trials"]["event"] = k[1]
    #         v.Xlabels["trials"]["bregion"] = k[2]
    #         v.Xlabels["trials"]["twind"] = [k[3] for _ in range(len(v.Xlabels["trials"]))]
    #
    #         tmp.append({
    #             "which_level":k[0],
    #             "event":k[1],
    #             "bregion":k[2],
    #             "twind":k[3],
    #             "pa":v
    #         })
    #     DFallpa = pd.DataFrame(tmp)
    #     out = DFallpa
    # else:
    #     out = DictEvBrTw_to_PA

    Params = {
        "animal":animal,
        "date":date,
        "files":files,
        "mult_or_single":mult_or_single
    }
    return DFallpa, Params


def dfpa_extract_single_window(DFallpa, which_level, event, twind):
    """ Return df with multiple pa (rows) all with the same specific
    values for wl, ev, and tw
    """
    a = DFallpa["which_level"] == which_level
    b = DFallpa["event"] == event
    c = DFallpa["twind"] == twind

    dfthis = DFallpa[(a & b & c)].reset_index(drop=True)
    assert len(dfthis)>0
    assert len(dfthis["bregion"].unique()) == len(dfthis)

    return dfthis

def dfpa_slice_specific_windows(DFallpa, list_pa_get):
    """ Return slice of DF which has onkly the specific onbinations of
    (wl, ev, tw) in list_pa_get
    PARAMS:
    - list_pa_get, list of tuples, each (wl, ev, tw), and thus each deefining a
    slice of DF (rows are bregions). Will collect then in order of items in list_pa_get
    Example:
    # list_pa_get = [
    #     ("stroke", "00_stroke", (-0.6, -0.5)),
    #     ("stroke", "00_stroke", (0.4, 0.6))
    # ]
    RETURNS:
    - df, concated slices, each one of the tuples of list_pa_get. N rows should be
    len(list_pa_get) * len(bregions). Will have same columnes as input df
    """
    list_df = []
    for wl, ev, tw in list_pa_get:
        list_df.append(dfpa_extract_single_window(DFallpa, wl, ev, tw))
    DFallpa_THIS = pd.concat(list_df).reset_index(drop=True)
    return DFallpa_THIS

def dfpa_group_and_split(DFallpa, vars_to_concat=None, vars_to_split=None,
                         DEBUG=False, concat_dim="trials"):
    """ Flexible method to concatenate PAs across all levels for
    given dimensions (vars_to_concat) and to maintain separate PA
    for each level of variables in vars_to_split.
    PARAMS:
    - vars_to_concat, list of str,
    - vars_to_split, list of str.
    (One of the above must be None, since they are redundant, just 2 methods
    to do same thing).
    RETURNS:
        DFallpa, with columns being the variables in vars_to_split, and each
        pa concated across vars_to_concat. For the columns which were grouped (concatted),
        replaces the value with "dummy", since the old values have been combined. THey
        are still accessible within the PA itself.
    EXAMPLE:
        vars_to_concat = ["which_level", "event", "twind"]
        vars_to_split = None
    """
    from neuralmonkey.classes.population import concatenate_popanals_flexible

    assert concat_dim in ["trials", "times"], "not coded yet"
    assert "bregion" in vars_to_split, "not sure how best to ###concat bregions, since they have diff chans..."

    allvars = ["which_level", "event", "bregion", "twind"]

    # They are redundant informations.
    if vars_to_concat is None:
        assert vars_to_split is not None
        vars_to_concat = [var for var in allvars if var not in vars_to_split]
    else:
        assert vars_to_split is None
        vars_to_split = [var for var in allvars if var not in vars_to_concat]

    # give a new conj var
    if "event" in vars_to_concat:
        # uiseful - a conjucntionv ariable for each tw
        from pythonlib.tools.pandastools import append_col_with_grp_index
        list_pa = DFallpa["pa"].tolist()
        list_pa_new = []
        for pa in list_pa:
            # print("HERERER", pa.Xlabels["trials"]["twind"].value_counts())
            pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"],
                                                            ["which_level", "event", "twind"],
                                                            "wl_ev_tw",
                                                            use_strings=True,
                                                            strings_compact=True)
            # print("HERERER", pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
        # assert False

    if False:
        for grp in DFallpa.groupby(vars_to_split):
            list_pa = grp[1]["pa"].tolist()
            # print(grp[1])

            # concatenate them
            pa_cat, twind_cat = concatenate_popanals_flexible(list_pa)
    else:
        def F(x):
            # concatenate them
            list_pa = x["pa"].tolist()
            return concatenate_popanals_flexible(list_pa, concat_dim=concat_dim)[0]

        DFallpa = DFallpa.groupby(vars_to_split, as_index=False).apply(F).reset_index(drop=True)
        # tmp = DFallpa.groupby(vars_to_split, as_index=False).apply(F)
        # # DFallpa = pd.DataFrame({"pa":tmp})
        # DFallpa = pd.DataFrame({"pa":tmp}, index=tmp.index)

    # For the other columns which were concated, they are not presnet in output.
    # add them back, with "dummy" value
    # print(vars_to_concat)
    # print(vars_to_split)
    for var in vars_to_concat:
        DFallpa[var] = "dummy"

    # HACKY, it returns df with column named None insted of pa.
    DFallpa["pa"] = DFallpa[None]
    del DFallpa[None]

    # HACKY - Redefine event to be conj varoiable. useful for downstream analy
    if "event" in vars_to_concat:
        for pa in DFallpa["pa"].tolist():
            # print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
            # assert False
            pa.Xlabels["trials"]["event"] = pa.Xlabels["trials"]["wl_ev_tw"]

            a = pa.X.shape[1]
            b = len(pa.Xlabels["trials"])
            c = max(pa.Xlabels["trials"].index)+1
            if not a==b==c:
                print(a, b, c)
                assert False

    assert len(DFallpa)>0

    if DEBUG:
        for pa in list_pa:
            print(pa.X.shape)

    if False: # No need to do this. This can fail for many legit reasions. E.g, diff events ahve diff not trials...
        # This is ok, since downstream will index into the dim values, and not just assume same shape
        # Sanity check that concated pas are identical across the concated dims
        from neuralmonkey.classes.population import check_get_common_values_this_dim
        list_pa = DFallpa["pa"].tolist()
        for dim in vars_to_concat:
            check_get_common_values_this_dim(list_pa, dim, assert_all_pa_have_same_values=True, dims_are_columns_in_xlabels=True)

    return DFallpa




    # ### [OBSOLETE] Devo - methods to combine acorss different events and time windows... (e.g., visual vs. motor...)
    # # Forst, for each PA, make sure it has info about event, twind, and bregion
    # for key, pa in DictEvBrTw_to_PA.items():
    #     ev, br, tw = key
    #     pa.Xlabels["trials"]["event"] = ev
    #     pa.Xlabels["trials"]["bregion"] = br
    #     n = len(pa.Xlabels["trials"])
    #     pa.Xlabels["trials"]["twind"] = [tw for _ in range(n)]
    #
    #     print(pa.X.shape)
    # from neuralmonkey.classes.population import concatenate_popanals
    #
    # # make sure all twinds are same length, with same time from alignment.
    # if False: # DONT NEED TO RUN. this was hacky solution. code is fixed.
    #     for k, pa in DictEvBrTw_to_PA.items():
    #         tw = k[2]
    #         pa = pa.slice_by_dim_values_wrapper("times", tw)
    #         DictEvBrTw_to_PA[k] = pa
    #
    # def convert_dict_pa_to_only_bregion_twind_keys(DictEvBrTw_to_PA, event):
    #     """
    #     """
    #     # collect all that have this event
    #     DictBregionTwindPA = {}
    #     for key, pa in DictEvBrTw_to_PA.items():
    #         ev, br, tw = key
    #         if ev==event:
    #             bregion_twind = (br, tw)
    #             assert bregion_twind not in DictBregionTwindPA.keys()
    #             DictBregionTwindPA[bregion_twind] = pa
    #     return DictBregionTwindPA
    #
    # convert_dict_pa_to_only_bregion_twind_keys(DictEvBrTw_to_PA, "03_samp")
    #
    # def _extract_concatted_pa(DictEvBrTw_to_PA, events=None, bregions=None, twinds=None):
    #     """
    #     Returns a single pa concatting. If any dimejsions are None, then takes
    #     all.
    #     NOTE: This concats across the pa dim of "trials". therefore must have same
    #     num chans and timepoints. This means, for now, can only pick one brain region.
    #     :param events:
    #     :param bregions:
    #     :param twinds:
    #     :return:
    #     """
    #
    #     assert isinstance(bregions, (list, tuple)) and len(bregions)==1, "see docs above."
    #
    #     list_pa = []
    #     list_twinds =[]
    #     # collect all that have this event
    #     for key, pa in DictEvBrTw_to_PA.items():
    #         ev, br, tw = key
    #         # bregion_twind = key[1:3]
    #         # event = key[0]
    #         if events is not None and ev not in events:
    #             continue
    #         if bregions is not None and br not in bregions:
    #             continue
    #         if twinds is not None and tw not in twinds:
    #             continue
    #
    #         assert pa.Xlabels["trials"]["event"].unique()[0]==ev, "sanity"
    #         list_pa.append(pa)
    #         list_twinds.append(tw)
    #
    #     # if you are combining multiple times, then replace times iwth a
    #     # dummy variable
    #     replace_times_with_dummy_variable = len(set(list_twinds))>1
    #
    #     assert len(list_pa)>0, "didnt get any data"
    #     return concatenate_popanals(list_pa, "trials",
    #                                 replace_times_with_dummy_variable=replace_times_with_dummy_variable)
    #
    # _extract_concatted_pa(DictEvBrTw_to_PA, events=None, bregions=["PMv_m"], twinds=None)
    #
    #
    # ##### Methods to collect specific slices across popanals
    # list_bregions = SP.bregion_list()
    # # Method 1 - pick specific conjunction of even and twind
    # list_events = ['03_samp', '06_on_strokeidx_0']
    # list_twinds = [(0.3, 0.5), (-0.3, -0.1)]
    # # list_twinds = [(0.1, 0.3), (0.1, 0.3)]
    # # list_events = ['04_go_cue', '06_on_strokeidx_0']
    # # list_twinds = [(-0.3, -0.1), (-0.1, 0.1)]
    #
    # for br in list_bregions:
    #
    #     # one pa for this bregion
    #     list_pa = []
    #     for ev, tw in zip(list_events, list_twinds):
    #         pa = _extract_concatted_pa(DictEvBrTw_to_PA, events=[ev], bregions=[br], twinds=[tw])
    #         assert pa.Xlabels["trials"]["event"].unique()[0]==ev, "sanity"
    #         list_pa.append(pa)
    #
    #     key = (br, (-99, 99))
    #     DictBregionTwindPA[key] = concatenate_popanals(list_pa, "trials", replace_times_with_dummy_variable=True)
    #
    #
    #
    # # Method 2 - ONLY WORKS if using same time window for each event
    #
    # # key_name_prune_event = False
    #
    # # For each bregion, twind, collect across all the events for it
    #
    # list_events_collect = ["03_samp", "06_on_strokeidx_0"]
    #
    # DictBregionTwindPA = {}
    # list_pa = []
    # # collect all that have this event
    # for key, pa in DictEvBrTw_to_PA.items():
    #     bregion_twind = key[1:3]
    #     event = key[0]
    #     if event in list_events_collect:
    #         if bregion_twind in DictBregionTwindPA.keys():
    #             DictBregionTwindPA[bregion_twind].append((event, pa))
    #         else:
    #             DictBregionTwindPA[bregion_twind] = [(event, pa)]
    #
    # # Concat
    # for br_tw, val in DictBregionTwindPA.items():
    #     list_ev = [v[0] for v in val]
    #     list_pa = [v[1] for v in val]
    #     assert list_ev==list_events_collect
    #
    #     # map_idxpa_to_value = {i:ev for i, ev in enumerate(list_ev)}
    #     map_idxpa_to_value = list_ev
    #     map_idxpa_to_value_colname = "event"
    #     DictBregionTwindPA[br_tw] = concatenate_popanals(list_pa, "trials",
    #                                                      map_idxpa_to_value=map_idxpa_to_value,
    #                                                      map_idxpa_to_value_colname=map_idxpa_to_value_colname)
    #
    #
    #
    # pa = DictBregionTwindPA[key]
    # pa.Xlabels["trials"]["event"].value_counts()
