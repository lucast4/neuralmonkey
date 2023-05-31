""" To collect analyzed data and make summary/overview/agg plots
"""


from pythonlib.tools.stringtools import decompose_string

# from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
# from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
# import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
from pythonlib.tools.expttools import fileparts, deconstruct_filename
import pickle
from pythonlib.tools.expttools import load_yaml_config
import glob
from neuralmonkey.classes.snippets import SAVEDIR_SNIPPETS_TRIAL
import pandas as pd
import pickle
from pythonlib.tools.pandastools import applyFunctionToAllRows
import numpy as np
from pythonlib.tools.expttools import writeDictToYaml

def load_and_preprocess_alldays(animal, ANALY_VER, var_desired, LIST_DATE,
    score_ver="r2_maxtime_1way_mshuff", which_level="trial"):
    """ Load all days into a single dataframe, and processing metadata along the way
    NOTE: will load if finds already saved data.
    """


    def _extract_var(var_str):
        """ Help read filename"""
        i = var_str.find("-OV")
        var = var_str[4:i]
        vars_conjunction_str = var_str[i+4:]
        vars_conjunction = decompose_string(vars_conjunction_str, "_") # assume underscore separates var

        print("var: ", var)
        print("vars_conjuction: ", vars_conjunction)
        return var, vars_conjunction, vars_conjunction_str

    def _preprocess_dfvar(DF_VAR, METADAT, _PARAMS):

        ################################# PREPROCESSING THINGS
        #### Go thru each row, map to semantic...
        def F(x):
            idx = x["META_idx"]
            event = x["event"]
            return _map_expt_event_toeventsemantic(DF_VAR, METADAT, idx, event)
        DF_VAR = applyFunctionToAllRows(DF_VAR, F, "eventsemantic")


        ## which var levels were present for each expt?
        for idx in METADAT.keys():
            levels = METADAT[idx]["var_levels"]
                
            list_epochkind = sorted(set([_get_epoch_kind(lev) for lev in levels]))

            # remove base
            list_epochkind = tuple([x for x in list_epochkind if not x=="base"])
            print(idx, ' .. ', levels, ' .. ', list_epochkind)
            
            # Store the kind
            METADAT[idx]["epochs_kind"] = tuple(list_epochkind)

        tmp = [METADAT[i]["epochs_kind"] for i in DF_VAR["META_idx"].tolist()]
        DF_VAR["epochs_kind"] = tmp

        # SAVE DF_VAR AGAIN
        # save the dataframe
        path = f"{SAVEDIR_ALL}/DF_VAR.pkl"
        with open(path, "wb") as f:
            pickle.dump(DF_VAR, f)
        print("Saved: ", path)

        _PARAMS["preprocess_done"]=True
        print("Saving _PARAMS to", f"{SAVEDIR_ALL}/PARAMS.yaml")
        writeDictToYaml(_PARAMS, f"{SAVEDIR_ALL}/PARAMS.yaml")  

        # save the metadat
        path = f"{SAVEDIR_ALL}/METADAT.pkl"
        with open(path, "wb") as f:
            pickle.dump(METADAT, f)
        print("Saved: ", path)

    ##############################
    if which_level=="trial":
        SAVEDIR_ALL = f"{SAVEDIR_SNIPPETS_TRIAL}/MULT_DAY/{animal}-{ANALY_VER}-{var_desired}-{score_ver}-{min(LIST_DATE)}-{max(LIST_DATE)}"
    else:
        assert False, "simple. make this dir"
    os.makedirs(SAVEDIR_ALL, exist_ok=True)
    print("DIR: ", SAVEDIR_ALL)

    # Try loading already saved
    if os.path.exists(f"{SAVEDIR_ALL}/DF_VAR.pkl"):
        with open(f"{SAVEDIR_ALL}/DF_VAR.pkl", "rb") as f:
            DF_VAR = pickle.load(f)

        _PARAMS = load_yaml_config(f"{SAVEDIR_ALL}/PARAMS.yaml")

        with open(f"{SAVEDIR_ALL}/METADAT.pkl", "rb") as f:
            METADAT = pickle.load(f)

        print("Loaded already-saved data!!", SAVEDIR_ALL)

        # if "preprocess_done" in _PARAMS.keys() and _PARAMS["preprocess_done"]==True:
        #     print("Skipping preprocess - already done!!")
        #     pass
        # else:
        print("Doing preprocess!!")
        _preprocess_dfvar(DF_VAR, METADAT, _PARAMS)

    else:

        ## Collect data across all days.
        LIST_DF_VAR = []
        idx = 0
        METADAT = {}

        for DATE in LIST_DATE:
            
            MS = load_mult_session_helper(DATE, animal)
            
            # make savedir
            sesses = "_".join([str(x) for x in list(range(len(MS.SessionsList)))])
            SAVEDIRTHIS = f"{SAVEDIR_SNIPPETS_TRIAL}/MULT_SESS/{MS.animal()}-{MS.date()}-{sesses}"

            # go thru each saved data
            sdir_check = f"{SAVEDIRTHIS}/{ANALY_VER}/var_by_varsother"
            list_dir = glob.glob(f"{sdir_check}/*")
            for _dir in list_dir:
                var_str = fileparts(_dir)[-2]
                var, vars_conjunction, vars_conjunction_str = _extract_var(var_str)

                if var==var_desired:

                    # 1) Load a single expt
                    sdir_base = f"{_dir}/SV_{score_ver}"
                    path = f"{sdir_base}/df_var.pkl"
                    try:
                        with open(path, "rb") as f:
                            df_var = pickle.load(f)
                        print("RELOADED df_var!!!")
                        print("... from:", path)
                        list_eventwindow_event = sorted(set([tuple(x) for x in df_var.loc[:, ["event", "_event"]].values.tolist()]))
                        df_var = df_var.reset_index(drop=True) # temporary fix.
                    except FileNotFoundError as err:
                        print("Could not find df_var (SKIPPING..):")
                        print(path)
                        continue                    

                    # Load params
                    path = f"{sdir_base}/Params.yaml"
                    Params = load_yaml_config(path)

                    path = f"{sdir_base}/ParamsGlobals.yaml"
                    ParamsGlobals = load_yaml_config(path)

                    path = f"{sdir_base}/params_to_save.yaml"
                    params_to_save = load_yaml_config(path)
                    
                    path = f"{sdir_base}/params_modulationgood_compute.yaml"
                    params_modulationgood_compute = load_yaml_config(path)

                    map_numevent_eventorig = {}
                    for numevent, eventorig in zip(Params["list_events_uniqnames"], Params["_list_events"]):
                        map_numevent_eventorig[numevent] = eventorig

                    # Get list of events and their dur and abstract_event names
                    map_eventname_params = {}
                    for event_window in params_modulationgood_compute["list_events_window"]:
                        numevent, eventabstract, pre_dur, post_dur = _eventwindow_to_predur_postdur(event_window)
                        map_eventname_params[event_window] = (eventabstract, numevent, pre_dur, post_dur)
                    
        #             for event, pre_dur, post_dur in zip(params_to_save["list_events"], 
        #                                                 params_to_save["list_pre_dur"],
        #                                                 params_to_save["list_post_dur"]):
        #                 event_window_combo_name = f"{event}_{pre_dur*1000:.0f}_to_{post_dur*1000:.0f}"
        #                 map_eventname_params[event_window_combo_name] = (map_numevent_eventorig[event], event, pre_dur, post_dur)
        #                 print("---")
        #                 print(event_window_combo_name, "-->")
        #                 print(map_numevent_eventorig[event], event, pre_dur, post_dur)
        #             print(map_eventname_params)    

                    ######################## get the correct vars conjucntion,
                    # (since the filename is too many uscores)
        #             vars_conjunction = df_var["var_others"].unique().tolist()[0]
                    vars_conjunction = params_modulationgood_compute["vars_conjuction"]
                    assert tuple(vars_conjunction) == df_var["var_others"].unique().tolist()[0]
                    assert len(df_var["var_others"].unique().tolist())==1
                    
                    ################ any same beh?
                    df_var = _extract_epochs_have_same_beh(df_var, vars_conjunction)

                    ########## Extract the levels for var (failed to save this in original anlayses)
                    if False:
                        # This is not necessary.
                        animal = MS.animal()
                        which_level = Params["which_level"]
                        from neuralmonkey.metadat.analy.anova_params import params_getter_plots, params_getter_extraction
                        from pythonlib.dataset.analy_dlist import concatDatasets
            #             params = params_getter_plots(animal, DATE, which_level, ANALY_VER)
                        paramsmeta = params_getter_extraction(animal, DATE, which_level, ANALY_VER)
                        list_dataset = []
                        for sn in MS.SessionsList:
                            D = sn.Datasetbeh

            #                 if paramsmeta["taskgroup_reassign_simple_neural"]:
            #                     # do here, so the new taskgroup can be used as a feature.
            #                     D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
            #                     print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
            #                     D.grouping_print_n_samples(["taskgroup", "probe"])

                            for this in paramsmeta["list_epoch_merge"]:
                                # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
                                D.supervision_epochs_merge_these(this[0], this[1], key=paramsmeta["epoch_merge_key"])

                            list_dataset.append(D)

                        # concat the datasets 
                        dataset_pruned_for_trial_analysis = concatDatasets(list_dataset)

                        var_levels = sorted(dataset_pruned_for_trial_analysis.Dat[var].unique().tolist())
                    else:
                        var_levels = []
                        for sn in MS.SessionsList:
                            var_levels.extend(sn.Datasetbeh.Dat[var].unique())
                        var_levels = sorted(set(var_levels))
                        
                    #### Append to dataframe
                    df_var["META_var"] = var
                    df_var["META_vars_conjunction"] = [tuple(vars_conjunction) for _ in range(len(df_var))]
                    # df_var["META_vars_conjunction_str"] = vars_conjunction_str
                    df_var["META_score_ver"] = score_ver
                    df_var["META_date"] = DATE
                    df_var["META_idx"] = idx
                    df_var["META_trial_or_block"] = _extract_trial_or_block(MS)

                    _params = {
                        "map_numevent_eventorig":map_numevent_eventorig,
                        "map_eventname_params":map_eventname_params,
                        "list_eventwindow_event":list_eventwindow_event,
                        "Params":Params,
                        "ParamsGlobals":ParamsGlobals,
                        "params_to_save":params_to_save,
                        "params_modulationgood_compute":params_modulationgood_compute,
                        "sdir_base":sdir_base,
                        "var_levels":var_levels,
                    }
                    METADAT[idx] = _params
                    
                    LIST_DF_VAR.append(df_var)
                    
                    idx+=1

        assert len(LIST_DF_VAR)==idx, "bug somewhere.."
        DF_VAR = pd.concat(LIST_DF_VAR).reset_index(drop=True)

        ###### SAVE
        # save params
        _PARAMS = {
            "LIST_DATE":LIST_DATE,
            "var_desired":var_desired,
            "ANALY_VER":ANALY_VER,
            "score_ver":score_ver,
            "animal":animal,
            "SAVEDIR_ALL":SAVEDIR_ALL
        }
        print("Saving _PARAMS to", f"{SAVEDIR_ALL}/PARAMS.yaml")
        writeDictToYaml(_PARAMS, f"{SAVEDIR_ALL}/PARAMS.yaml")    

        # save the dataframe
        path = f"{SAVEDIR_ALL}/DF_VAR.pkl"
        with open(path, "wb") as f:
            pickle.dump(DF_VAR, f)
        print("Saved: ", path)

        # save the metadat
        path = f"{SAVEDIR_ALL}/METADAT.pkl"
        with open(path, "wb") as f:
            pickle.dump(METADAT, f)
        print("Saved: ", path)

        ######## preprocess
        _preprocess_dfvar(DF_VAR, METADAT, _PARAMS)

    return DF_VAR, METADAT, _PARAMS


def _get_epoch_kind(level):
    if level in ["U", "D", "L", "L|0", "R", "TR", "TR|0"]:
        return "dir"
    elif level in ["VlL1", "llV1", "llV1|0", "llV1R"]:
        return "shape"
    elif level in ["AnBmTR", "AnBmTR|0", "AnBm1b"]:
        return "AnBm"
    elif level in ["TR|1", "llV1|1", "L|1", "AnBmTR|1", "rndstr"]:
        return "rankcol"
    elif level in ["base"]:
        return "base"
    else:
        print(level)
        assert False, "add this"

def _extract_trial_or_block_inner(epochs, DEBUG=False, MIN_BLOCK_NTRIALS=15,
                           PLOT=False):
    """ Inner code for _extract_trial_or_block, see docs there
    PARAMS
    - epochs, list of int or str
    """

    # Get transitions across adjacent trials.   
    epochs = ["__START__"] + [x for x in epochs] + ["__END__"] # append entry into the first trial.
    epochs1 = epochs[:-1]
    epochs2 = epochs[1:]
    transitions_sameness = np.asarray([e1==e2 for e1, e2 in zip(epochs1, epochs2)], dtype=int)

    nsame = sum(transitions_sameness) # how many transitions were between same epochs?
    ntot = len(transitions_sameness)-2 # minus endpoints
    # e.g, [1,0,1,0,1,1,1,0, 0, 1] --> (3,9)
    
    # Segmenmtation - find trnasitions, then n and len of blocks of same epochs.
    # transitions_inds = np.where(np.diff(np.asarray(transitions_sameness, dtype=int))==1) # entries into blocks.

    # transitions_inds = np.where(np.abs(np.diff(transitions_sameness)==1)) # entries into blocks.
    transitions_inds = np.where(transitions_sameness==0) # entries into blocks.

    # print("HERE")
    # print(transitions_sameness)
    # print(np.diff(np.asarray(transitions_sameness, dtype=int)))
    # print(transitions_inds)
    # assert False
    block_lengths = np.diff(transitions_inds)
    # get all blocks longer than N trials
    block_lengths_actual = block_lengths[block_lengths>=MIN_BLOCK_NTRIALS]
    n_blocks = len(block_lengths_actual)

    if PLOT:
        # plot boolean aray of block transition times
        plt.figure()
        plt.plot(transitions, '-ok')
        print("block_lengths_actual:", block_lengths_actual)
    
    if n_blocks==0:
        # a single block...
        nblocks = 1
        mean_len_blocks = len(epochs1)
    else:
        mean_len_blocks = np.mean(block_lengths_actual)    

    
    if DEBUG:
        print("transitions_sameness: ", transitions_sameness)
        print("INdices of transitions between blocks: ", transitions_inds)
        print("n pairs of adjavent trials that are same epoch: ", nsame)
        print("total pairs:", ntot)
        print("n blocks found: ", n_blocks)
        print("block_lengths_actual:", block_lengths_actual)
        print("mean len of blocks: ", mean_len_blocks)
        for e1, e2 in zip(epochs1, epochs2):
            print(e1, e2)

    if nsame/ntot<0.65:
        trial_or_block = "trial"
#     elif nsame/ntot>0.72 and n_blocks>4 and mean_len_blocks<100:
    elif nsame/ntot>0.72 and n_blocks>4:
        trial_or_block = "blockfast"
    elif nsame/ntot>0.72 and n_blocks<=4 and mean_len_blocks>50:
        trial_or_block = "blockslow"
    else:
        print(nsame)
        print(ntot)
        print(nsame/ntot)
        print(n_blocks)
        print(mean_len_blocks)
        assert False, "which is it?"
    
    return trial_or_block


def _extract_trial_or_block(MS, DEBUG=False, MIN_BLOCK_NTRIALS=15,
                           PLOT=False):
    """ Return string (trial or slowblock, fastblock)
    which version?
    PARAMS;
    - MIN_BLOCK_NTRIALS, int, only calls something a block if it has
    >= this many consecutive trials of same epoch
    """
    import numpy as np
    
    D = MS.datasetbeh_extract()
    epochs = D.Dat["epoch"].tolist()
    return _extract_trial_or_block_inner(epochs, DEBUG, MIN_BLOCK_NTRIALS, PLOT)


def _extract_epochs_have_same_beh(df_var, vars_conjunction, call_first_stroke_matched_the_same=False):
    """ Is the same behavior done across all epochs? Infer based on the othervar, and heuristics.
    PARAMS;
    - vars_conjunction, list/tuple of str, the conjunctive vars used.
    - call_first_stroke_matched_the_same, bool, if True, then cases with first stroke matched will
    call "same" otherwise call diff.
    RETURNS:
    - df_var, with new column: is_same_beh_across_epochs
    """
    if "taskgroup" in vars_conjunction:
        def F(x):
            if "same" in x["taskgroup"]:
                return True
            else:
                return False
    elif "epochset" in vars_conjunction:
        def F(x):
            # if epochset has all the epochs, then it is same
            epochset = x["epochset"]
            if isinstance(epochset, tuple) and len(epochset)>1:
                print("Hacky, should actually check that this epochset has all the epochs")
                return True
            elif isinstance(epochset, tuple) and len(epochset)==1:
                return False
            elif isinstance(epochset, str) and epochset=="LEFTOVER":
                return False
            else:
                print(epochset)
                assert False
    elif all([x in vars_conjunction for x in ['seqc_0_loc', 'seqc_0_shape', 'seqc_nstrokes_beh']]):
        def F(x):
            # then the first stroke is matched. 
            return call_first_stroke_matched_the_same
    else:
        print(vars_conjunction)
        assert False

    df_var = applyFunctionToAllRows(df_var, F, "is_same_beh_across_epochs")
    print("Appended column to df_var: is_same_beh_across_epochs")

    return df_var

def _eventwindow_sort(list_eventwindow):
    """ Sort chronotically
    PARAMS;
    - list_eventwindow, list of str, either:
    --- 06_on_strokeidx_0_-250_to_350 or
    --- 00_baseline
    """

    list_keys = []
    for ev in list_eventwindow:
        numev, _, predur, postdur = _eventwindow_to_predur_postdur(ev)
        if predur is not None:
            key = (numev, predur, postdur, ev)
        else:
            key = (numev, ev, ev, ev)
        list_keys.append(key)

    list_keys = sorted(list_keys)

    list_eventwindow_sorted = [x[3] for x in list_keys]
    return list_eventwindow_sorted


def _eventwindow_to_predur_postdur(eventwindow):
    """ Convert from string holding the specific event and its time winodw
    to decomposed data, inmcludiong predur/postdur
    PARAMS:
    - eventwindow, e.g, eventwindow = "05_first_raise_-600_to_-50" or
    00_baseline, in which case predur and postdur returns NOne.
    RETURNS:
    - (numevent, eventabstract, pre_dur, post_dur)
    e.g., ('05_first_raise', 'first_raise', -0.6, -0.05)
    """
    
    tmp = decompose_string(eventwindow, "_") # ['05', 'first', 'raise', '-600', 'to', '-50']


    if len(tmp)<5:
        # ['05', 'first', 'raise']
        pre_dur = None
        post_dur = None
        # print(eventwindow)
        # print(tmp)
        # assert False
        numevent = "_".join(tmp)
        assert numevent==eventwindow, f"sanioty check..., {numevent}, {eventwindow}"
        eventabstract = "_".join(tmp[1:])
    else:
        # ['05', 'first', 'raise', '-600', 'to', '-50']
        pre_dur = int(tmp[-3])/1000
        post_dur = int(tmp[-1])/1000
        numevent = "_".join(tmp[:-3])
        eventabstract = "_".join(tmp[1:-3])
        
    return numevent, eventabstract, pre_dur, post_dur

def _extract_list_event(DF_VAR, idx):
    """ RETurn list of str that are the events fro this expt.
    returns things like 00_...
    """
    return sorted(DF_VAR["event"].unique())

def _extract_uses_rulecue2(DF_VAR, idx):
    """ from all the events for this day, decide if there are is rulecue
    RETURNS bool
    """
    list_event = _extract_list_event(DF_VAR, idx)
    for ev in list_event:
        _, eventabstract, _, _= _eventwindow_to_predur_postdur(ev)
        if eventabstract=="rulecue2":
            return True
    return False
    # for eventparams in METADAT[idx]["map_eventname_params"].values():
    #     if eventparams[0]=="rulecue2":
    #         return True
    # return False

# Which kind of trial structure? e.g., rule cues?
def _extract_trial_structure_kind(DATE):
    """ Hard coded, trial structure (e.g.,w hat cues) for each day
    """
    if (DATE in [220812, 220814, 220815, 220816, 220827, 220913, 220921, 220928, 220929, 220930]) or (DATE in [221001]):
        # v0: # fixcue[nocolor] --> fixtouch --> image --> go...
        return "v0_default"
    elif DATE in [221002, 221014, 221020, 221021, 221023, 221024]:
        # v1: # fixcue[colored] --> fixtouch --> image[colored] --> go...
        return "v1_colorfix"
    elif DATE in [221031, 221102, 221107, 221112, 221113, 221114, 221118, 221119, 221121, 221125]:
        # v2: # fuxcue[nocolor] --> fixtouch --> rulecue2[e.g, fixcue_color_change] --> samp + cue_color_off
        return "v2_cue2"
    else:
        print(DATE)
        assert False


MAP_EVENT_EVENTCODESEMANTIC = {} # MAP_EVENT_EVENTCODESEMANTIC[trial_structure_kind][("samp", -1, -1)]:"baseline",
MAP_EVENT_EVENTCODESEMANTIC["v0_default"] = {
        ("samp", -1, -1):"baseline",
        ("samp", 1, 1):"imageE",
        ("go_cue", -1, -1):"imageL",
        ("first_raise", -1, -1):"firstraise",
        ("on_strokeidx_0", -1, 1):"stroke",
        ("doneb", -1, 1):"doneb",
        ("post", 1, 1):"post",
        ("reward", 1, 1):"reward",        
        ("reward_all", 1, 1):"reward"        
    }
MAP_EVENT_EVENTCODESEMANTIC["v1_colorfix"] = {
        ("fixcue", -1, -1):"baseline",
        ("fixcue", 1, 1):"rulecueE",
        ("samp", -1, -1):"rulecueL",
        ("samp", 1, 1):"imageE",
        ("go_cue", -1, -1):"imageL",
        ("first_raise", -1, -1):"firstraise",
        ("on_strokeidx_0", -1, 1):"stroke",
        ("doneb", -1, 1):"doneb",
        ("post", 1, 1):"post",
        ("reward", 1, 1):"reward",        
        ("reward_all", 1, 1):"reward"        
    }
MAP_EVENT_EVENTCODESEMANTIC["v2_cue2"] = {
        ("rulecue2", -1, -1):"baseline",
        ("rulecue2", 1, 1):"rulecueE",
        ("samp", -1, -1):"rulecueL",       
        ("samp", 1, 1):"imageE",
        ("go_cue", -1, -1):"imageL",
        ("first_raise", -1, -1):"firstraise",
        ("on_strokeidx_0", -1, 1):"stroke",
        ("doneb", -1, 1):"doneb",
        ("post", 1, 1):"post",
        ("reward", 1, 1):"reward",        
        ("reward_all", 1, 1):"reward"        
    }
CACHE_EVENTSEMANTIC={}
def _map_expt_event_toeventsemantic(DF_VAR, METADAT, idx, event):
    """ Given an expt idx and event, map this to a string that is the semantic
    code for the evetn (e/g. "baseline")
    """
    if (idx, event) not in CACHE_EVENTSEMANTIC.keys():
        
        _, eventabstract, predur, postdur = _eventwindow_to_predur_postdur(event)
        # eventabstract, _, predur, postdur = METADAT[idx]["map_eventname_params"][event]
        uses_rulecue2 = _extract_uses_rulecue2(DF_VAR, idx)
        
        # - which trial structure
        tmp = DF_VAR[DF_VAR["META_idx"]==idx]["META_date"].unique().tolist()
        if len(tmp)!=1:
            print(tmp)
            print(DF_VAR["META_idx"].value_counts())
            print(DF_VAR["META_date"].value_counts())
            assert False
        date = tmp[0]
        trial_structure_kind = _extract_trial_structure_kind(date)

        # Get the event semantic.
        eventcode = (eventabstract, int(predur/abs(predur)), int(postdur/abs(postdur)))
        map_eventcode_eventsemantic = MAP_EVENT_EVENTCODESEMANTIC[trial_structure_kind]

        try:
            eventsemantic = map_eventcode_eventsemantic[eventcode]
        except Exception as err:
            print(idx)
            print(eventcode)
            print(map_eventcode_eventsemantic)
            raise err
        if False:
            print(eventcode, '... mapped to eventsemantic ... ', eventsemantic)
        
        CACHE_EVENTSEMANTIC[(idx, event)] = eventsemantic
    
    return CACHE_EVENTSEMANTIC[(idx, event)]

def _plot(dfthis, sdir_base):
    """ Helper to make all basic plots for this dataframe
    """
    if len(dfthis)==0:
        return

    dfthis = dfthis.copy()
    eventsemantic_ordered = _plot_get_eventsemantic_ordered(dfthis)
    
    # condition df to be compatible with common plots.
    dfthis["var_others"] = [("DUMMY",) for _ in range(len(dfthis))]

    # give the events starting indices
    map_ev_numev = {}
    for i, ev in enumerate(eventsemantic_ordered):
        if i<10:
            map_ev_numev[ev] = f"0{i}_{ev}"
        else:
            map_ev_numev[ev] = f"{i}_{ev}"

    def F(x):
        return map_ev_numev[x["eventsemantic"]]
    dfthis = applyFunctionToAllRows(dfthis, F, "event")

    dfthis["_event"] = dfthis["event"]

    from neuralmonkey.classes.snippets import Snippets
    sdir = f"{sdir_base}/modulation"
    os.makedirs(sdir, exist_ok=True)
    SP = Snippets(None, None, None, None, None, None, None, SKIP_DATA_EXTRACTION=True)
    
    # Modualtion plots
    print("Plotting modulation")
    SP.modulationgood_plot_summarystats(dfthis, savedir=sdir, skip_agg=True)

    plt.close("all")

    # print("** Plotting heatmaps")
    # sdir = f"{sdir_base}/modulation_heatmap"
    # os.makedirs(sdir, exist_ok=True)
    # print(sdir)
    sdir = f"{sdir_base}/modulation_heatmap"
    os.makedirs(sdir, exist_ok=True)
    print("Plotting brain schematic.")
    SP.modulationgood_plot_brainschematic(dfthis, sdir=sdir) 

    plt.close("all")


def _plot_get_eventsemantic_ordered(DF_VAR):
    """ Return semantic events that exist, ordered by hand in 
    chronoligical order for plotting
    """
    # eventsemantic_ordered = sorted(dfthis["eventsemantic"].unique())
    
    # print(sorted(DF_VAR["eventsemantic"].unique()))
    eventsemantic_ordered = ['baseline',
     'rulecueE',
     'rulecueL',
     'imageE',
     'imageL',
     'firstraise',
     'stroke',
     'doneb',
     'post',
     'reward',
     ]
    return eventsemantic_ordered

def plot_all(DF_VAR, METADAT, _PARAMS):
    """
    HElper to make all plots
    """

    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel

    DF_VAR_AGG, DF_VAR_AGG_EXPT = aggregate_df_var(DF_VAR)

    SAVEDIR_ALL = _PARAMS["SAVEDIR_ALL"]
    eventsemantic_ordered = _plot_get_eventsemantic_ordered(DF_VAR)

    # suammrize conjucntions of variables in entire dataset
    from pythonlib.tools.pandastools import grouping_print_n_samples
    # path = f"{sdir}/data_groups_overview.txt"
    path = f"{SAVEDIR_ALL}/groupings_overview.txt"
    grouping_print_n_samples(DF_VAR_AGG, ["epochs_kind", "META_trial_or_block", "is_same_beh_across_epochs", "META_idx", "META_date", "META_vars_conjunction"], savepath=path, save_as="txt");
        

    ##### [good], plot aggregate, each row is a combination of desired indices
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # DF_VAR = append_col_with_grp_index(DF_VAR, ["epochs_kind", "META_trial_or_block", "is_same_beh_across_epochs"], "_rowplot")
    # DF_VAR_AGG = append_col_with_grp_index(DF_VAR_AGG, ["META_trial_or_block", "is_same_beh_across_epochs"], "_rowplot")
    # DF_VAR_AGG["_rowplot"].value_counts()

    # How to split into rows.
    list_grouping = [
        ["META_trial_or_block", "is_same_beh_across_epochs"],
        ["epochs_kind", "META_trial_or_block", "is_same_beh_across_epochs"],
    ]

    # Plot
    for level in ["chan", "expt"]:
        if level=="chan":
            dfthis = DF_VAR_AGG.copy()
        elif level=="expt":
            dfthis = DF_VAR_AGG_EXPT.copy()
        else:
            print(level)
            assert False

        for i, grouping in enumerate(list_grouping):
            dfthis = append_col_with_grp_index(dfthis, grouping, "_rowplot")

            if level=="chan":
                kinds_to_plot = ["point", "strip"]
            elif level=="expt":
                kinds_to_plot = ["point", "strip"]
            else:
                assert False

            for kind in kinds_to_plot:
                
                for row in ["_rowplot", None]:
                    print("Plotting ... ", level, grouping, kind)
                
                    if kind=="bar":
                        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", row=row, order=eventsemantic_ordered, kind="bar", ci=68)
                    elif kind=="strip":
                        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", row=row, order=eventsemantic_ordered, jitter=True, alpha=0.25)
                    elif kind=="point":
                        if row=="_rowplot":
                            fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", row="epochs_kind", hue="_rowplot", order=eventsemantic_ordered, kind="point", ci=68)
                        elif row is None:
                            fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue="_rowplot", order=eventsemantic_ordered, kind="point", ci=68)
                        else:
                            print(hue)
                            assert False
        # #                 fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", row="META_trial_or_block", hue="_rowplot", order=eventsemantic_ordered, kind="point", ci=68)
        #                 fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue="_rowplot", order=eventsemantic_ordered, kind="point", ci=68)
        #                 rotateLabel(fig)
        #                 for ax in fig.axes.flatten():
        #                     ax.axhline(0, color="k", alpha=0.4)                        
        #                 fig.savefig(f"{SAVEDIR_ALL}/overview-{level}-grouping_{i}-{kind}_simple.pdf")
                    else:
                        print(kind)
                        assert False

                    rotateLabel(fig)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.4)                        
                    fig.savefig(f"{SAVEDIR_ALL}/overview-{level}-grouping_{i}-{kind}-row_{row}.pdf")

                    plt.close("all")

    # A single line for each "expt" (i.e., group by _rowplot)
    from pythonlib.tools.snstools import plotgood_lineplot
    dfthis = DF_VAR_AGG_EXPT.copy()
    dfthis = append_col_with_grp_index(dfthis, ["META_trial_or_block", "is_same_beh_across_epochs"], "_rowplot")
    for include_scatter in [True, False]:
        # for val_kind in df_var_chan["val_kind"].unique():
        #     df_var_chan_this = df_var_chan[df_var_chan["val_kind"]==val_kind]
        fig = plotgood_lineplot(dfthis, xval="eventsemantic", yval="val", line_grouping="_rowplot",
                                include_mean=False, include_scatter=include_scatter,
                                relplot_kw={"col":"bregion"});

        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.25)
        rotateLabel(fig)

        fig.savefig(f"{SAVEDIR_ALL}/overview-lines-scatter_{include_scatter}.pdf")

        plt.close("all")


    ##### Plots, split by various things
    # ALL
    sdir = f"{SAVEDIR_ALL}"
    os.makedirs(sdir, exist_ok=True)
    dfthis = DF_VAR_AGG
    _plot(dfthis, sdir)

    # trial vs. block
    for tb in ["trial", "block"]:
        sdir = f"{SAVEDIR_ALL}/META_trial_or_block-{tb}"
        os.makedirs(sdir, exist_ok=True)
        dfthis = DF_VAR_AGG[DF_VAR_AGG["META_trial_or_block"]==tb]
        _plot(dfthis, sdir)

    # Epoch kind
    list_epochs_kind = set([v["epochs_kind"] for v in METADAT.values()])
    list_epochs_kind

    for ek in list_epochs_kind: # e./.g,  ('dir', 'shape')
        idxs = [i for i, v in enumerate(METADAT.values()) if v["epochs_kind"]==ek]
        
        # Pull out these expts
        dfthis = DF_VAR_AGG[DF_VAR_AGG["META_idx"].isin(idxs)]
        
        # savedir
        sdir = f"{SAVEDIR_ALL}/epochs_kind-{'|'.join(ek)}"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        
        # Plot
        _plot(dfthis, sdir)       

    # same beh vs. diff

    for samebeh in [True, False]:
    #     dfthis = DF_VAR_AGG[(DF_VAR_AGG["META_trial_or_block"]==tb) & (DF_VAR_AGG["is_same_beh_across_epochs"]==samebeh)]
        dfthis = DF_VAR_AGG[(DF_VAR_AGG["is_same_beh_across_epochs"]==samebeh)]
        sdir = f"{SAVEDIR_ALL}/is_same_beh_across_epochs-{samebeh}"
        os.makedirs(sdir, exist_ok=True)
        dfthis = DF_VAR_AGG[DF_VAR_AGG["META_trial_or_block"]==tb]
        _plot(dfthis, sdir)


    # Combo of (same beh) and (trial/block)
    for samebeh in [True, False]:
        for tb in ["trial", "block"]:
        
            dfthis = DF_VAR_AGG[(DF_VAR_AGG["META_trial_or_block"]==tb) & (DF_VAR_AGG["is_same_beh_across_epochs"]==samebeh)]
            sdir = f"{SAVEDIR_ALL}/same_beh-{samebeh}-trialblock-{tb}"
            os.makedirs(sdir, exist_ok=True)
            print("This many datpts for, ", samebeh, tb, len(dfthis))
            _plot(dfthis, sdir)


    # For each day, plot, separating by "var others"
    list_date = DF_VAR["META_date"].unique().tolist()
    for date in list_date:
        
        sdir = f"{SAVEDIR_ALL}/{date}"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        dfthis = DF_VAR[(DF_VAR["META_date"]==date)].copy()
    #     dfthis["event"] = dfthis["eventsemantic"]
    #     order = sorted(dfthis["event"].unique())
        
        # Save metadat
        from pythonlib.tools.pandastools import grouping_print_n_samples
        path = f"{sdir}/data_groups_overview.txt"
        grouping_print_n_samples(dfthis, ["META_idx", "META_date", "META_trial_or_block", "is_same_beh_across_epochs", "META_vars_conjunction"], savepath=path, save_as="txt")
        
        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", kind="point", ci=68, hue="is_same_beh_across_epochs",
                         row="var_others", order=eventsemantic_ordered)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.25)
        from pythonlib.tools.snstools import rotateLabel
        rotateLabel(fig)
        fig.savefig(f"{sdir}/bars_split_by_varothers.pdf")

        dfthis = append_col_with_grp_index(dfthis.copy(), ["var_others", "is_same_beh_across_epochs"], "_rowplot")    
        
        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", row="_rowplot", order=eventsemantic_ordered,
                         alpha=0.25, jitter=True)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.25)
        from pythonlib.tools.snstools import rotateLabel
        rotateLabel(fig)
        fig.savefig(f"{sdir}/scatter_split_by_varothers.pdf")                

        plt.close("all")


def aggregate_df_var(DF_VAR):
    """ Get all aggregated versions
    """

    # aggregate so that each eventsemantic has single datapt
    from pythonlib.tools.pandastools import aggregGeneral

    # conjunction of everything except values or what want to agg (event)
    # Agg over (1) all lower-events, keeping just the semantic event. and (2) all var_others, splitting just by whether is same beh
    grouping = ['chan',
     'var',
    #  'var_others',
    #  'lev_in_var_others',
     'val_kind',
     'val_method',
     'bregion',
     'is_same_beh_across_epochs',
     'META_var',
     'META_vars_conjunction',
     'META_score_ver',
     'META_date',
     'META_idx',
     'META_trial_or_block',
     'eventsemantic',
       'epochs_kind']
    if "exptgrp" in DF_VAR.columns:
        grouping.append("exptgrp")
    # grouping = [col for col in DF_VAR.columns if col not in ["_event", "event", "val", "val_zscore", "n_datapts"]]
    DF_VAR_AGG = aggregGeneral(DF_VAR, group = grouping, values = ["val", "val_zscore", "n_datapts"]) # one datapt is one chan

    # Same as above, but each datapt is an expt, agging over chans.
    grouping = [
     'var',
    #  'var_others',
    #  'lev_in_var_others',
     'val_kind',
     'val_method',
     'bregion',
     'is_same_beh_across_epochs',
     'META_var',
     'META_vars_conjunction',
     'META_score_ver',
     'META_date',
     'META_idx',
     'META_trial_or_block',
     'eventsemantic',
       'epochs_kind']
    if "exptgrp" in DF_VAR.columns:
        grouping.append("exptgrp")
    DF_VAR_AGG_EXPT = aggregGeneral(DF_VAR, group = grouping, values = ["val", "val_zscore", "n_datapts"]) # one datapt is one expt


    return DF_VAR_AGG, DF_VAR_AGG_EXPT


def plotwrapper_contrasts(DF_VAR, var_contrast, grouping, _PARAMS, PRINT=False,
    SAVE_SUFFIX = None, MINIMAL_PLOTS=False):
    """ Make plots that emphasize the contrast between levels for var_contrast.
    PARAMS:
    - var_contrast, string, column in DF_VAR,..
    - grouping, list of str, these are the "expt" level datapts
    - MINIMAL_PLOTS, then (i) skips scatter plots (takes a while) and some other
    plots ofo conjucntion of variables.
    """

    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from neuralmonkey.classes.session import REGIONS_IN_ORDER 
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.expttools import writeDictToYaml

    ## Extract params
    grouping = [g for g in grouping if g!=var_contrast]
    levels = DF_VAR[var_contrast].unique().tolist()

    ## Condition dataset
    DF_VAR = append_col_with_grp_index(DF_VAR, grouping, "exptgrp", use_strings=False)
    DF_VAR_AGG, DF_VAR_AGG_EXPT = aggregate_df_var(DF_VAR)
    # extract expts that have all levels of var_contrast
    n_min=1
    if PRINT:
        print(DF_VAR_AGG["META_trial_or_block"].value_counts())
        print(DF_VAR_AGG["exptgrp"].value_counts())
        print(var_contrast)
        print(levels)
        grouping_print_n_samples(DF_VAR_AGG, ["exptgrp", "META_trial_or_block"])

    DF_VAR_AGG, _ = extract_with_levels_of_conjunction_vars(DF_VAR_AGG, var_contrast, ["exptgrp"], levels, n_min, 
        lenient_allow_data_if_has_n_levels=2, PRINT=PRINT)
    DF_VAR_AGG_EXPT, _ = extract_with_levels_of_conjunction_vars(DF_VAR_AGG_EXPT, var_contrast, ["exptgrp"], levels, n_min, 
        lenient_allow_data_if_has_n_levels=2, PRINT=PRINT)

    ## Plots
    eventsemantic_ordered = _plot_get_eventsemantic_ordered(DF_VAR)
    tmp = DF_VAR["bregion"].unique().tolist()
    REGIONS_IN_ORDER = [r for r in REGIONS_IN_ORDER if r in tmp]

    if False:
        # in progress -- need to fix the code for computing differences
        # for each chan, take difference (if 2 levels)
        ############# difference
        from pythonlib.tools.pandastools import summarize_featurediff
        INDEX = ["exptgrp", "bregion", "chan", "eventsemantic", "META_date", "is_same_beh_across_epochs", "var_others", "lev_in_var_others"]
        for x in grouping:
            if x not in INDEX:
                INDEX.append(x)        
        INDEX = [x for x in INDEX if x!=var_contrast]
        INDEX = ["exptgrp", "chan", "eventsemantic", "bregion", "META_date", "var_others", "lev_in_var_others"]

        dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(DF_VAR, 
                                                                                                               GROUPING=var_contrast, 
                                                                                                               GROUPING_LEVELS=levels,
                             FEATURE_NAMES=["val"], INDEX=INDEX, return_dfpivot=True)

        fig = sns.catplot(data=dfsummary, x="eventsemantic", y="val-ruleswERRORminrulesw", col="bregion",
                      order = eventorder, row="grp", kind="point", ci=68, aspect=1)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.25)
        fig.savefig(f"{sdir}/{date}_error_minus_rules.pdf")
        plt.close("all")

    ## SAVEDIR
    if SAVE_SUFFIX:
        SAVEDIR = f"{_PARAMS['SAVEDIR_ALL']}/PLOTS_CONTRAST_{var_contrast}_{SAVE_SUFFIX}"
    else:
        SAVEDIR = f"{_PARAMS['SAVEDIR_ALL']}/PLOTS_CONTRAST_{var_contrast}"
    os.makedirs(SAVEDIR, exist_ok=True)

    ## Save text, params.
    path = f"{SAVEDIR}/grouping_datchan_DF_VAR_AGG.txt"
    grouping_print_n_samples(DF_VAR_AGG, grouping+[var_contrast], savepath=path, save_as="txt")

    path = f"{SAVEDIR}/grouping_datexpt_DF_VAR_AGG_EXPT.txt"
    grouping_print_n_samples(DF_VAR_AGG_EXPT, grouping+[var_contrast], savepath=path, save_as="txt")

    path = f"{SAVEDIR}/params.yaml"
    _params = {
        "var_contrast":var_contrast,
        "grouping":grouping
    }
    writeDictToYaml(_params, path )
    
    # Grand average | combine levels | dat=chan
    fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", order=eventsemantic_ordered, 
        kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)             
      
    fig.savefig(f"{SAVEDIR}/grand-combined-dat_chan.pdf")
    plt.close("all")

    # Grand average | combine levels | dat=expt
    fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", order=eventsemantic_ordered, 
        kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)             
      
    fig.savefig(f"{SAVEDIR}/grand-combined-dat_expt.pdf")
    plt.close("all")

    # Grand average | dat=chan
    fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                      order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)             
      
    fig.savefig(f"{SAVEDIR}/grand-dat_chan.pdf")
    plt.close("all")

    # Grand average | dat=expt
    fig = sns.catplot(data=DF_VAR_AGG_EXPT, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                      order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)                        
    fig.savefig(f"{SAVEDIR}/grand-dat_expt.pdf")
    plt.close("all")

    # Row = level | data = mean of chan
    fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue="exptgrp", 
                      row=var_contrast,
                      order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)                        
    # fig.savefig(f"{SAVEDIR_ALL}/overview-{level}-grouping_{i}-{kind}-row_{row}.pdf")

    fig.savefig(f"{SAVEDIR}/row_level-dat_chan-point.pdf")
    plt.close("all")

    # Row = level | data = mean of expt
    fig = sns.catplot(data=DF_VAR_AGG_EXPT, x="eventsemantic", y="val", col="bregion", hue="exptgrp", 
                      row=var_contrast,
                      order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.4)                        
    fig.savefig(f"{SAVEDIR}/row_level-dat_expt-point.pdf")

    if not MINIMAL_PLOTS:
        # Row = level | data = scatter chan
        fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue="exptgrp", 
                          row=var_contrast,
                          order=eventsemantic_ordered, jitter=True, alpha=0.4, col_order=REGIONS_IN_ORDER)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.4)                        
        fig.savefig(f"{SAVEDIR}/row_level-dat_chan-scatter.pdf")
        plt.close("all")

    # row = exptgrp | {mean_chan, scatter_chan}
    ngrps = len(DF_VAR_AGG["exptgrp"].unique())
    if ngrps<10:
        fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                          row="exptgrp",
                          order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.4)                           
        fig.savefig(f"{SAVEDIR}/row_exptgrp-dat_chan-point.pdf")
        
        if not MINIMAL_PLOTS:
            fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                              row="exptgrp",
                              order=eventsemantic_ordered, jitter=True, alpha=0.4, col_order=REGIONS_IN_ORDER)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.4)                        
            fig.savefig(f"{SAVEDIR}/row_exptgrp-dat_chan-scatter.pdf")

        # fig.savefig(f"{SAVEDIR_ALL}/overview-{level}-grouping_{i}-{kind}-row_{row}.pdf")
        plt.close("all")
        
        

    # Each row = levels for other marginals
    for _row in ["META_trial_or_block", "epochs_kind", "is_same_beh_across_epochs"]:
        if not _row==var_contrast:
            
            fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                              row=_row,
                              order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.4)                        
            fig.savefig(f"{SAVEDIR}/row_{_row}-dat_chan-point.pdf")
            
            
        if not MINIMAL_PLOTS:
            fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                              row=_row,
                              order=eventsemantic_ordered, jitter=True, alpha=0.35, col_order=REGIONS_IN_ORDER)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.4)                        
            fig.savefig(f"{SAVEDIR}/row_{_row}-dat_chan-scatter.pdf")
                
            plt.close("all")

            
            if False:
                # Too crowded, not useful
                fig = sns.catplot(data=DF_VAR_AGG, x="eventsemantic", y="val", col="bregion", hue=var_contrast, 
                                  row=_row,
                                  order=eventsemantic_ordered, jitter=True, alpha=0.3, col_order=REGIONS_IN_ORDER)
                rotateLabel(fig)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.4)                        


        #     ### Each row = exptgrp | datapt = chans
        #     dfthis = DF_VAR_AGG_EXPT[DF_VAR_AGG_EXPT["exptgrp"]==grp]
        #     fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue=var_contrast,
        #                       order=eventsemantic_ordered, jitter=True, alpha=0.3, col_order=REGIONS_IN_ORDER)
        #     rotateLabel(fig)
        #     for ax in fig.axes.flatten():
        #         ax.axhline(0, color="k", alpha=0.4)                        

            # One plot for each level
            _levels = DF_VAR_AGG[_row].unique()
            for lev in _levels:
                dfhis = DF_VAR_AGG[DF_VAR_AGG[_row]==lev]
                fig = sns.catplot(data=dfhis, x="eventsemantic", y="val", col="bregion", hue="exptgrp", 
                                  row=var_contrast,
                                  order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
                rotateLabel(fig)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.4)                        
                fig.savefig(f"{SAVEDIR}/row_{var_contrast}-lev_of_{_row}_{lev}-dat_chan.pdf")
            plt.close("all")

                    

    # Fig = exptgrp | Row = level | data = chan
    list_grp = DF_VAR_AGG["exptgrp"].unique()
    for grp in list_grp:
        dfthis = DF_VAR_AGG[DF_VAR_AGG["exptgrp"]==grp]

        if not MINIMAL_PLOTS:
            fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue=var_contrast,
                              order=eventsemantic_ordered, jitter=True, alpha=0.3, col_order=REGIONS_IN_ORDER)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.4)                        
            fig.savefig(f"{SAVEDIR}/row_{var_contrast}-exptgrp_{grp}-dat_chan-scatter.pdf")
            
        
        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue=var_contrast,
                          order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.4)                        
        fig.savefig(f"{SAVEDIR}/row_{var_contrast}-exptgrp_{grp}-dat_chan-point.pdf")
        plt.close("all")
        
        fig = sns.catplot(data=dfthis, x="eventsemantic", y="val", col="bregion", hue=var_contrast,
                            row=var_contrast,
                          order=eventsemantic_ordered, kind="point", ci=68, col_order=REGIONS_IN_ORDER)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.4)                        
        fig.savefig(f"{SAVEDIR}/row_{var_contrast}-exptgrp_{grp}-dat_chan-point-2.pdf")
        plt.close("all")
        

    if False:
        # too large, and not useful, since events are split into subplots..
        # for each chan, plot its comparison between levels

        from pythonlib.tools.snstools import plotgood_lineplot
        # plotgood_lineplot(dfthisthis, xval=var_contrast, yval="val", line_grouping="chan")
        # plotgood_lineplot(dfthis, xval=var_contrast, yval="val", line_grouping="exptgrp")
        fig = plotgood_lineplot(dfthis, xval=var_contrast, yval="val", line_grouping="exptgrp", rowvar="bregion",
                         colvar="eventsemantic")