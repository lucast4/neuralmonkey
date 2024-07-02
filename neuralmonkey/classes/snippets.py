import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.listtools import sort_mixed_type
import os
import seaborn as sns
from pythonlib.tools.exceptions import NotEnoughDataException
import pickle
from pythonlib.tools.plottools import savefig
from pythonlib.tools.snstools import rotateLabel

SAVEDIR_SNIPPETS_STROKE = "/gorilla1/analyses/recordings/main/anova/bystroke" # for snippets
SAVEDIR_SNIPPETS_TRIAL = "/gorilla1/analyses/recordings/main/anova/bytrial" # for snippets

# LIST_SUPERV_NOT_TRAINING = ["off|0||0", "off|1|solid|0", "off|1|rank|0"] # Dont use this. use D.preprocessGood(no supervision) insetead

# import warnings
# warnings.filterwarnings("error")

# def load_snippets(sdir, fname="Snippets"):
#     import pickle as pkl
#
#     path = f"{sdir}/{fname}.pkl"
#     with open(path, "rb") as f:
#         SP = pkl.load(f)
#
#     return SP
#

def load_snippet_single(sn, which_level):
    """ Load a single prevsiouly saved snippet
    """

    sess = sn.RecSession

    # 1) initialize SP from SN, without data
    sp = Snippets(sn, None, None, None, None, None, None, SKIP_DATA_EXTRACTION=True)

    # 2) load
    SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/by{which_level}"
    sdir = f"{SAVEDIR}/{sn.Animal}-{sn.Date}-sess_{sess}"

    # 3) regenerate all data.
    sp.load_v2(sdir)
    
    # within this sp, save mapping to its session
    if "session_idx" in sp.DfScalar.columns:
        assert np.all(sp.DfScalar["session_idx"]==sess), "new version extracts session_idx directly, while old version loads it"

    sp.DfScalar["session_idx"] = sess

    sp.datamod_append_outliers()

    return sp

def concat_mult_snippets(list_sp, MS, SITES_COMBINE_METHODS = "intersect",
    DEBUG = False, prune_low_fr_sites=True):
    """
    Concatenate snippets that represent multiple sessions from a single day, where
    list_sp could be reloaded or extyracted.

    3/17/24 - Deleteing sp.DS if this is strokes version, becuase some features in DS are
    from binning with dataset, and these bins would be unrelated across sp. Therefore
    should regenerate DS.
    """
    from pythonlib.tools.checktools import check_objects_identical

    if False:
        tmp = list(set([sp.Params["which_level"] for sp in list_sp]))
        assert len(tmp)==1
        which_level = tmp[0]

    print("This many vals across loaded session")
    for i, sp in enumerate(list_sp):

        print(i, ":", len(sp.DfScalar))

        sn = sp.SN

        # Track its origin
        sp.DfScalar["session_neural"] = sn.RecSession
        assert i==sn.RecSession, "confusing... not necessary but best to fix this so not confused later."

    # # list_session = [0,1,2,3]
    # list_sp = []
    # # list_sn = []
    # for i, sn in enumerate(MS.SessionsList):
    #
    #     sess = sn.RecSession
    #     assert i==sess, "confusing... not necessary but best to fix this so not confused later."
    #
    #     sp = load_snippet_single(sn, which_level)
    #
    #     # Track its origin
    #     sp.DfScalar["session_neural"] = i
    #
    #     # store
    #     list_sp.append(sp)
    #     # list_sn.append(sn)

    # # 4) concatenate all sessions.
    # for i, sp in enumerate(list_sp):
    #     print(i, ":", len(sp.DfScalar))

    # Use the first session
    # sn = MS.session_generate_single_nodata()
    # sn = MS.SessionsList[0]

    # 1) initialize SP from SN, without data
    SPall = Snippets(None, None, None,
                     None, None,
                     None, None, SKIP_DATA_EXTRACTION=True)
    SPall.DfScalar = pd.concat([sp.DfScalar for sp in list_sp]).reset_index(drop=True)

    # Other stuff
    # SPall.SN = None
    # SPall.ListPA = None
    # SPall.DS = None
    SPall._CONCATTED_SNIPPETS = True
    SPall.SNmult = MS

    # Deleted DS, and regenrate, so that tbinned variables are using entire Datast
    # New version, where you don't load old snippets.
    if True:
        for sp in list_sp:
            sp.DS = None
        SPall.DS = None
    else:
        # do concat once here
        # This has problems if do computation in D, should then extract DS from D
        from pythonlib.dataset.dataset_strokes import concat_dataset_strokes
        list_ds = [sp.DS for sp in list_sp]
        DS = concat_dataset_strokes(list_ds)
        SPall.DS = DS
    # else:
    #     # Old, but this means needs to concat each time read it.
    #     SPall.DSmult = [sp.DS for sp in list_sp]

    # ind_sess = 0
    # trial_within = 0 
    # ms.index_convert((ind_sess, trial_within))

    # get params that are same for all sp
    DEBUG = False
    SPall._CONCATTED_SNIPPETS = True

    # 0) Check that DfScalar has same columns
    cols_prev = None
    for sp in list_sp:
        cols_this = set(sp.DfScalar.columns)
        if cols_prev is not None:
            if cols_prev != cols_this:
                print(cols_prev)
                print(cols_this)
                assert False, "must be identical. you should reextract raw"
        cols_prev = cols_this

    # 1) check they are ideitncal across sp
    list_attr_identical = ["Params", "ParamsGlobals"]
    for attr in list_attr_identical:
        items = [getattr(sp, attr) for sp in list_sp]
        for i in range(len(items)):
            for j in range(i+1, len(items)):
    #             print(i, j)
                if not DEBUG:
                    assert check_objects_identical(items[i], items[j], PRINT=True)
        item_take = items[0]
        print(f"Assigning to SP.{attr} this item:")
        print(item_take)
        setattr(SPall, attr, item_take)

    # Assign attributes that might be different across sp.
    list_attr_union = ["Sites"]
    for attr in list_attr_union:
        items = [getattr(sp, attr) for sp in list_sp]
        if SITES_COMBINE_METHODS == "union":
            items_flatten = [x for it in items for x in it]
            print(items_flatten)
            assert False, "confirm correct then remove this."
            items_combine = list(set(items_flatten))
        elif SITES_COMBINE_METHODS == "intersect":
            items_combine = list(set.intersection(*[set(x) for x in items]))
        else:
            print(SITES_COMBINE_METHODS)
            assert False
        setattr(SPall, attr, items_combine)
        # print("items: ", len(items[0]), len(items[1]), SITES_COMBINE_METHODS, ":", len(items_combine))

    SPall.Sites = sorted(SPall.Sites)

    # In data, keep only the sites in self.Sites
    SPall.DfScalar = SPall.DfScalar[SPall.DfScalar["chan"].isin(SPall.Sites)].reset_index(drop=True)

    # Remove columns from DfScalar that are ambiguous
    # TODO...

    # make sure is numbered
    SPall.DfScalar["event"] = SPall.DfScalar["event_aligned"]

    # Cleanups of DS
    if SPall.DS is not None:
        SPall.DS.clean_preprocess_if_reloaded()

    # Only keep sites with high fr
    if prune_low_fr_sites:
        SPall.prune_low_firing_rate_sites()

        # In case pruned
        SPall.Sites = sorted(SPall.DfScalar["chan"].unique().tolist())

    return SPall

def load_and_concat_mult_snippets(MS, which_level, events_keep, SITES_COMBINE_METHODS = "intersect",
    DEBUG = False, prune_low_fr_sites=True, REGENERATE_SNIPPETS=True, PRE_DUR=None, POST_DUR=None):
    """ [GOOD] For both loading pre-saved Snippets and regenerating new Snippets.
    previously saved snippets using save_v2
    PARAMS:
    - MS, MultSessions, holding all the sessions for which each will
    load a single snippet.
    - SITES_COMBINE_METHODS, str, when combining attribtues that are lists, across 
    sessions, how to deal if they are not same? 
    --- intersect, union
    - SAVEDIR, has subdirs like SAVEDIR/amimal_date_sess/DfScalar.pkl
    RETURNS:
    - SPall, concatted SP
    - SAVEDIR_ALL, a newly genreated path
    """

    if REGENERATE_SNIPPETS:
        from neuralmonkey.scripts.analy_snippets_extract import extract_snippets_all_sessions
        # EVENTS_KEEP = None
        # which_level = "trial"
        # EVENTS_KEEP = ["03_samp", "go_cue"]
        list_sp = extract_snippets_all_sessions(MS, which_level, events_keep, 1, False, PRE_DUR=PRE_DUR, POST_DUR=POST_DUR)
        SAVEDIR_ALL = None
    else:
        import os

        SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/by{which_level}"

        # Genreate the seave dir.
        # Assumes there is only single animal and date... (not necessary, just
        # conven ient for filenames)
        sesses = "_".join([str(x) for x in list(range(len(MS.SessionsList)))])
        SAVEDIR_ALL = f"{SAVEDIR}/MULT_SESS/{MS.animal()}-{MS.date()}-{sesses}"
        # if DEBUG:
        #     SAVEDIR_ALL = SAVEDIR_ALL + "-DEBUG"
        os.makedirs(SAVEDIR_ALL, exist_ok=True)

        # list_session = [0,1,2,3]
        list_sp = []
        # list_sn = []
        for i, sn in enumerate(MS.SessionsList):

            sess = sn.RecSession
            assert i==sess, "confusing... not necessary but best to fix this so not confused later."

            sp = load_snippet_single(sn, which_level)

            # # Track its origin
            # sp.DfScalar["session_neural"] = i

            # store
            list_sp.append(sp)
            # list_sn.append(sn)

    SPall = concat_mult_snippets(list_sp, MS, SITES_COMBINE_METHODS,
                                              DEBUG, prune_low_fr_sites)

    return SPall, SAVEDIR_ALL

    # # 4) concatenate all sessions.
    # print("This many vals across loaded session")
    # for i, sp in enumerate(list_sp):
    #     print(i, ":", len(sp.DfScalar))
    #
    # # Use the first session
    # # sn = MS.session_generate_single_nodata()
    # # sn = MS.SessionsList[0]
    #
    # # 1) initialize SP from SN, without data
    # SPall = Snippets(None, None, None,
    #                  None, None,
    #                  None, None, SKIP_DATA_EXTRACTION=True)
    # SPall.DfScalar = pd.concat([sp.DfScalar for sp in list_sp]).reset_index(drop=True)
    #
    # # Other stuff
    # # SPall.SN = None
    # # SPall.ListPA = None
    # # SPall.DS = None
    # SPall._CONCATTED_SNIPPETS = True
    # SPall.SNmult = MS
    # if True:
    #     # do concat once here
    #     from pythonlib.dataset.dataset_strokes import concat_dataset_strokes
    #     list_ds = [sp.DS for sp in list_sp]
    #     DS = concat_dataset_strokes(list_ds)
    #     SPall.DS = DS
    # else:
    #     # Old, but this means needs to concat each time read it.
    #     SPall.DSmult = [sp.DS for sp in list_sp]
    #
    # # ind_sess = 0
    # # trial_within = 0
    # # ms.index_convert((ind_sess, trial_within))
    #
    # # get params that are same for all sp
    # DEBUG = False
    # SPall._CONCATTED_SNIPPETS = True
    #
    # # 0) Check that DfScalar has same columns
    # cols_prev = None
    # for sp in list_sp:
    #     cols_this = set(sp.DfScalar.columns)
    #     if cols_prev is not None:
    #         if cols_prev != cols_this:
    #             print(cols_prev)
    #             print(cols_this)
    #             assert False, "must be identical. you should reextract raw"
    #     cols_prev = cols_this
    #
    #
    # # 1) check they are ideitncal across sp
    # list_attr_identical = ["Params", "ParamsGlobals"]
    # for attr in list_attr_identical:
    #     items = [getattr(sp, attr) for sp in list_sp]
    #     for i in range(len(items)):
    #         for j in range(i+1, len(items)):
    # #             print(i, j)
    #             if not DEBUG:
    #                 assert check_objects_identical(items[i], items[j], PRINT=True)
    #     item_take = items[0]
    #     print(f"Assigning to SP.{attr} this item:")
    #     print(item_take)
    #     setattr(SPall, attr, item_take)
    #
    # # Assign attributes that might be different across sp.
    # list_attr_union = ["Sites"]
    # for attr in list_attr_union:
    #     items = [getattr(sp, attr) for sp in list_sp]
    #     if SITES_COMBINE_METHODS == "union":
    #         items_flatten = [x for it in items for x in it]
    #         print(items_flatten)
    #         assert False, "confirm correct then remove this."
    #         items_combine = list(set(items_flatten))
    #     elif SITES_COMBINE_METHODS == "intersect":
    #         items_combine = list(set.intersection(*[set(x) for x in items]))
    #     else:
    #         print(SITES_COMBINE_METHODS)
    #         assert False
    #     setattr(SPall, attr, items_combine)
    #     # print("items: ", len(items[0]), len(items[1]), SITES_COMBINE_METHODS, ":", len(items_combine))
    #
    # SPall.Sites = sorted(SPall.Sites)
    #
    # # In data, keep only the sites in self.Sites
    # SPall.DfScalar = SPall.DfScalar[SPall.DfScalar["chan"].isin(SPall.Sites)].reset_index(drop=True)
    #
    # # Remove columns from DfScalar that are ambiguous
    # # TODO...
    #
    # # make sure is numbered
    # SPall.DfScalar["event"] = SPall.DfScalar["event_aligned"]
    #
    # # Cleanups of DS
    # if SPall.DS is not None:
    #     SPall.DS.clean_preprocess_if_reloaded()
    #
    # # Only keep sites with high fr
    # if prune_low_fr_sites:
    #     SPall.prune_low_firing_rate_sites()
    #
    # return SPall, SAVEDIR_ALL


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
        prune_feature_levels_min_n_trials=None, 
        dataset_pruned_for_trial_analysis=None,
        trials_prune_just_those_including_events=True,
        fr_which_version="sqrt",
        NEW_VERSION=True,
        SKIP_DATA_EXTRACTION =False,
        fail_if_times_outside_existing=True,
        DS_pruned = None
        ):
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
        be used to determine which trials (i.e. onluy tjhose in Dataset). NOTE: used only for trials,
        not for its data.            
        NOTE: see extraction_helper() for notes on params.
        """


        self.DfScalar = None
        self.DfScalarBeforeRemoveSuperv = None
        self.DfScalarBeforePrune = None
        self.SN = SN
        self.SNmult = None
        # self.DSmult = None
        self._NEW_VERSION = NEW_VERSION
        self._SKIP_DATA_EXTRACTION = SKIP_DATA_EXTRACTION
        self._CONCATTED_SNIPPETS = False
        self._LOADED = False
        self.ListPA = None
        self.DS = None
        # To cache sanity checks.
        self._SanityFrSmTimesIdentical = None

        if fail_if_times_outside_existing==False:
            assert which_level in ["trial", "flex"], "only coded for this level... just add it below."
            
        if SKIP_DATA_EXTRACTION:
            # Then useful if tyou want to load old data.
            return

        sites = SN.sitegetterKS_map_region_to_sites_MULTREG()
        # 1b) Which sites to use?
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

        if dataset_pruned_for_trial_analysis is None:
            dataset_pruned_for_trial_analysis = SN.Datasetbeh

        if False:
            # Stop doing this, to allow different durations
            if NEW_VERSION:
                assert len(list(set(list_pre_dur)))==1, "assumes same. prob makes sense, then later prune for computation."
                assert len(list(set(list_post_dur)))==1, "assumes same. prob makes sense, then later prune for computation."
                PRE_DUR = list_pre_dur[0]
                POST_DUR = list_post_dur[0]

        ### EXTRACT SNIPPETS
        if which_level in ["stroke", "stroke_off"]:
            # Sanity checks
            assert len(list_events)==0, "event is stroke. you made a mistake (old code, site anova?)"
            assert len(list_pre_dur)==1
            assert len(list_post_dur)==1
            events_that_must_include = None

            pre_dur = list_pre_dur[0]
            post_dur = list_post_dur[0]

            # Each datapt matches a single stroke
            if False:
                if DS_pruned is None:
                    assert False, "dont extract here, isntead, extract AFTER load params"
                    # Then get it. Otherwise just use the input and assume you did everthing right.
                    DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, "clean_one_to_one",
                        strokes_only_keep_single, tasks_only_keep_these,
                        None,
                        list_features_extraction)
                else:
                    DS = DS_pruned
            else:
                # Just get DS without any pruning.
                DS = datasetstrokes_extract(dataset_pruned_for_trial_analysis, "all_no_clean")

            # Filter the trials
            trials = SN.get_trials_list(True, True, only_if_in_dataset=True,
                dataset_input=dataset_pruned_for_trial_analysis,
                events_that_must_include = events_that_must_include)
            print("\n == extracting these trials: ", trials)
            trialcodes = [SN.datasetbeh_trial_to_trialcode(t) for t in trials]
            DS.dataset_prune_by_trialcodes(trialcodes)

            print("Extracting, SN.snippets_extract_bystroke...")
            for var in list_features_extraction:
                if (var not in SN.Datasetbeh.Dat.columns) and (var not in DS.Dat.columns):
                    print(var)
                    print(SN.Datasetbeh.Dat.columns)
                    print(DS.Dat.columns)
                    assert False, "get this feature"

            if which_level=="stroke":
                align_to="onset"
            elif which_level=="stroke_off":
                align_to="offset"
            else:
                assert False

            if NEW_VERSION:
                DfScalar = SN.snippets_extract_bystroke(sites_keep, DS, 
                    features_to_get_extra=list_features_extraction, 
                    fr_which_version=fr_which_version, pre_dur=pre_dur,
                    post_dur=post_dur,
                    align_to=align_to)
                ListPA = None
            else:
                assert False, "this is incorrect"
                ListPA = self.extract_snippets_strokes(DS, sites_keep, pre_dur, post_dur,
                    features_to_get_extra=list_features_extraction)


            # Fill in dummy variables
            # list_events = ["stroke"]
            # list_events_uniqnames = ["00_stroke"]

            DfScalar["event"] = "00_stroke"
            # NOTE: DfScalar["event_aligned"] will be "00_stroke"

            list_events = DfScalar["event"].unique().tolist()
            list_events_uniqnames = DfScalar["event_aligned"].unique().tolist()

        elif which_level in ["substroke", "substroke_off"]:
            # Sanity checks
            assert len(list_events)==0, "event is stroke. you made a mistake (old code, site anova?)"
            assert len(list_pre_dur)==1
            assert len(list_post_dur)==1
            events_that_must_include = None

            pre_dur = list_pre_dur[0]
            post_dur = list_post_dur[0]

            # Add features specific to substrokes
            features_for_substrokes = ["shape", "index_within_stroke", 
                                       "circularity_binned", "distcum_binned", "angle_binned",
                                       "dist_angle"]
            list_features_extraction = list_features_extraction + features_for_substrokes

            # Extract all substrokes
            if False:
                print("*** RUNNING PIPELINE TO GET SUBSTROKES...")
                from pythonlib.dataset.substrokes import pipeline_wrapper
                D, DS, _ = pipeline_wrapper(dataset_pruned_for_trial_analysis)
            else:
                # Instead of running pipeline, run previously adn sthen load here. This
                # important -- combines across sessions for computing, which reduces noise a
                # lot.
                from pythonlib.dataset.substrokes import load_presaved_using_pipeline, features_motor_extract_and_bin
                DS, D = load_presaved_using_pipeline(dataset_pruned_for_trial_analysis)

            # Replace DatasetBeh in SN, needed for downstream code.
            # And clear things that store event-related stuff, since this
            # might include strokes.
            SN.Datasetbeh = D
            SN.EventsTimeUsingPhd = {}
            # SN.PopAnalDict = {}
            SN._CachedStrokesPeanutsOnly = {}
            SN._CachedStrokes = {}

            # Filter the trials
            trials = SN.get_trials_list(True, True, only_if_in_dataset=True,
                dataset_input=dataset_pruned_for_trial_analysis,
                events_that_must_include = events_that_must_include)
            print("\n == extracting these trials: ", trials)
            trialcodes = [SN.datasetbeh_trial_to_trialcode(t) for t in trials]
            DS.Dat = DS.Dat[DS.Dat["trialcode"].isin(trialcodes)].reset_index(drop=True)

            for var in list_features_extraction:
                if (var not in SN.Datasetbeh.Dat.columns) and (var not in DS.Dat.columns):
                    print(var)
                    print(SN.Datasetbeh.Dat.columns)
                    print(DS.Dat.columns)
                    assert False, "get this feature"

            if which_level=="substroke":
                align_to="onset"
            elif which_level=="substroke_off":
                align_to="offset"
            else:
                assert False

            print("Extracting, SN.snippets_extract_bysubstroke...")
            DfScalar = SN.snippets_extract_bysubstroke(sites_keep, DS,
                features_to_get_extra=list_features_extraction, pre_dur=pre_dur,
                post_dur=post_dur,
                align_to=align_to)
            ListPA = None

            DfScalar["event"] = "00_substrk"
            DfScalar["event_aligned"] = "00_substrk"
            list_events = DfScalar["event"].unique().tolist()
            list_events_uniqnames = DfScalar["event_aligned"].unique().tolist()

        elif which_level=="trial":
            # Each datapt is a single trial
            # no need to extract antyhing, use sn.Datasetbeh
            # only those trials that exist in SN.Datasetbeh

            # trials = SN.get_trials_list(True, True, only_if_in_dataset=True, 
            #     dataset_input=dataset_pruned_for_trial_analysis,
            #     events_that_must_include = list_events)
            assert trials_prune_just_those_including_events==True, "this on by defualt. if turn off, then change line below in SN.get_trials_list"
            events_that_must_include = ["fix_touch", "on_strokeidx_0"]
            trials = SN.get_trials_list(True, True, only_if_in_dataset=True,
                dataset_input=dataset_pruned_for_trial_analysis,
                events_that_must_include = events_that_must_include)
            print("\n == extracting these trials: ", trials)
            DS = None

            assert len(list_events) == len(list_pre_dur)
            assert len(list_events) == len(list_post_dur)

            if NEW_VERSION:
                # Extract snippets across all trials, sites, and events.
                # trials = sn.get_trials_list(True)[:5]
                # list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
                #     'stroke_index', 'stroke_index_fromlast', 'stroke_index_semantic', 
                #     'shape_oriented', 'ind_taskstroke_orig', 'gridloc',
                #     'gridloc_x', 'gridloc_y', 'h_v_move_from_prev']

                # Fail here if dataset doesnt have features
                for var in list_features_extraction:
                    if var not in SN.Datasetbeh.Dat.columns:
                        print(var)
                        print(SN.Datasetbeh.columns)
                        assert False, "get this feature"

                if True:
                    # Use flex method. THis should be identical, except (i) faster, and (ii) if multipel instances of event exists in
                    # a trial, this gets all (previously only got the first), and (iii) the former can use diff pre/postdur for each
                    # event, wheras now uses a single one..
                    assert len(set(list_pre_dur))==1, "see not above."
                    assert len(set(list_post_dur))==1
                    pre_dur = list_pre_dur[0]
                    post_dur = list_post_dur[0]

                    DfScalar = SN.snippets_extract_by_event_flexible(sites_keep, trials,
                        list_events, pre_dur, post_dur,
                        features_to_get_extra=None, fr_which_version="sqrt", DEBUG=False,
                        fail_if_times_outside_existing=fail_if_times_outside_existing)

                    # Fill in dummy variables
                    DfScalar["event"] = DfScalar["event_unique_name"]  # "00_go"
                    DfScalar["event_aligned"] = DfScalar["event_unique_name"]  # "00_go"
                    list_events_uniqnames = sorted(list(DfScalar["event"].unique()))
                else:
                    DfScalar, list_events_uniqnames = SN.snippets_extract_bytrial(sites_keep, trials,
                        list_events, list_pre_dur, list_post_dur,
                        features_to_get_extra=list_features_extraction)
                    DfScalar["event"] = DfScalar["event_aligned"]

                ListPA = None
            else:
                ListPA, list_events_uniqnames = self.extract_snippets_trials(trials, sites_keep, list_events, list_pre_dur, list_post_dur,
                    list_features_extraction)


        elif which_level == "flex":
            # Flexible, based on event markers (abstract, or timestamp, can work).

            # Sanity checks
            assert len(list_events)>0, "must have events to align to"
            assert len(list_features_extraction)==0, "better to label each datapt with faeture AFTER extraction."
            assert len(list_features_get_conjunction)==0, "better to label each datapt with faeture AFTER extraction."
            assert strokes_only_keep_single==False, "leave this False, it is not relevant for flex"
            assert tasks_only_keep_these is None, "not yet coded"
            assert prune_feature_levels_min_n_trials is None, "not yet coded"
            assert trials_prune_just_those_including_events ==False, "not yet coded"
            # assert dataset_pruned_for_trial_analysis is None, "not yet coded"
            assert NEW_VERSION==True

            assert len(list_pre_dur)==1, "not yet coded for diff pre_due for each event"
            assert len(list_post_dur)==1
            pre_dur = list_pre_dur[0]
            post_dur = list_post_dur[0]

            if trials_prune_just_those_including_events:
                events_that_must_include = list_events
            else:
                events_that_must_include = None
            trials = SN.get_trials_list(True, True, only_if_in_dataset=True, 
                dataset_input=dataset_pruned_for_trial_analysis,
                events_that_must_include = events_that_must_include)
            print("\n == extracting these trials: ", trials)

            # print("Extracting, SN.snippets_extract_bystroke...")
            # for var in list_features_extraction:
            #     if (var not in SN.Datasetbeh.Dat.columns) and (var not in DS.Dat.columns):
            #         print(var)
            #         print(SN.Datasetbeh.Dat.columns)
            #         print(DS.Dat.columns)
            #         assert False, "get this feature"

            DfScalar = SN.snippets_extract_by_event_flexible(sites_keep, trials,
                list_events, pre_dur, post_dur, 
                features_to_get_extra=None, fr_which_version="sqrt", DEBUG=False,
                fail_if_times_outside_existing=fail_if_times_outside_existing)
            ListPA = None
            DS = None
            # Fill in dummy variables
            # list_events = ["stroke"]
            list_events_uniqnames = SN.events_rename_with_ordered_index(list_events)
            # e.g, ['00_go', '01_doneb', '02_reward_all']

            # Replace events_aligned with the num_event.
            # THis is assumed for many of codes downstream.
            DfScalar["event"] = DfScalar["event_unique_name"]  # "00_go"
            # DfScalar["event_aligned"] = DfScalar["event_unique_name"]  # "00_go"
        else:
            assert False

        # Map var conjunctions.
        if NEW_VERSION:
            map_var_to_othervars = None
        else:
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
            # "list_events":list_events,
            "_list_events":list_events,
            "list_events_uniqnames":list_events_uniqnames,
            "list_features_extraction":list_features_extraction,
            "list_features_get_conjunction":list_features_get_conjunction,
            "list_pre_dur":list_pre_dur,
            "list_post_dur":list_post_dur,
            "map_var_to_othervars":map_var_to_othervars,
            "strokes_only_keep_single":strokes_only_keep_single,
            "tasks_only_keep_these":tasks_only_keep_these,
            "prune_feature_levels_min_n_trials":prune_feature_levels_min_n_trials,
            "fr_which_version":fr_which_version,
            "SPIKES_VERSION":SN.SPIKES_VERSION
        }
        self.globals_initialize()

        # Genreate scalars
        if NEW_VERSION:
            self.DfScalar = DfScalar
            # get this, for use with removing outliers.
            self.DfScalar = self.datamod_compute_fr_scalar(self.DfScalar)
            # SKIP, not using it. can compute on fly.
            self.DfScalar["fr_sm_sqrt"] = self.DfScalar["fr_sm"]**0.5
            self.DfScalar["session_idx"] = SN.RecSession
        else:
            assert False, "dont use"
            if False:
                # Old version
                print("\n == listpa_convert_to_scalars")
                self.listpa_convert_to_scalars()
                print("\n == pascal_convert_to_dataframe")
                self.pascal_convert_to_dataframe(fr_which_version=fr_which_version)
            else:
                self.listpa_convert_to_scalar_v2(fr_which_version=fr_which_version)
                self.DfScalar = self.DfScalar # they are the same

        # Make sure you save bregions in dfscalar
        self.datamod_append_bregion(self.DfScalar)

        # Get useful variables
        if not NEW_VERSION:
            print("\n == _preprocess_map_features_to_levels")        
            self._preprocess_map_features_to_levels()
            self._preprocess_map_features_to_levels_input("character")

        if not NEW_VERSION:
            map_var_to_levels = {}
            for var in self.Params["list_features_get_conjunction"]:
                map_var_to_levels[var] = sort_mixed_type(self.DfScalar[var].unique().tolist())
        else:
            map_var_to_levels = None
        self.Params["map_var_to_levels"] = map_var_to_levels

        print(f"** Generated Snippets, (ver {which_level}). Final length of SP.DfScalar: {len(self.DfScalar)}")

        # self.DfScalar["event"] = self.DfScalar["event_aligned"]
        # 1/4/23 - used to do it, but this takes lot of space t
        if False:
            # only do this if needed
            self.datamod_remove_outliers()

    def globals_initialize(self):
        """ Initialize self.ParamsGlobals to defaults"""
        self.ParamsGlobals = {
            "n_min_trials_per_level":5,
            "lenient_allow_data_if_has_n_levels":2,
            "PRE_DUR_CALC":self.Params["list_pre_dur"][0],
            "POST_DUR_CALC":self.Params["list_post_dur"][0],
            "list_events":self.Params["list_events_uniqnames"],
            "list_pre_dur":self.Params["list_pre_dur"],
            "list_post_dur":self.Params["list_post_dur"]
        }

    def globals_update(self, 
            globals_nmin = None,
            globals_lenient_allow_data_if_has_n_levels = None,
            PRE_DUR_CALC=None, 
            POST_DUR_CALC=None,
            list_events = None,
            list_pre_dur = None,
            list_post_dur = None,
            PRINT=True):
        """ 
        Update self.ParamsGlobals specific keys (those that are not None).
        PArams for ongoing anaklysis,.
        NOTE: Leave anything None to skip update.
        """

        if globals_nmin is not None:
            self.ParamsGlobals["n_min_trials_per_level"] = globals_nmin
        if globals_lenient_allow_data_if_has_n_levels is not None:
            self.ParamsGlobals["lenient_allow_data_if_has_n_levels"] = globals_lenient_allow_data_if_has_n_levels
        if PRE_DUR_CALC is not None:
            self.ParamsGlobals["PRE_DUR_CALC"] = PRE_DUR_CALC
        if POST_DUR_CALC is not None:
            self.ParamsGlobals["POST_DUR_CALC"] = POST_DUR_CALC
        if list_events is not None:
            for ev in list_events:
                if ev not in self.Params["list_events_uniqnames"]:
                    print(ev)
                    print(self.Params["list_events_uniqnames"])
                    assert False, "You entered an incorrect event name"
            self.ParamsGlobals["list_events"] = list_events
        if list_pre_dur is not None:
            self.ParamsGlobals["list_pre_dur"] = list_pre_dur
        if list_post_dur is not None:
            self.ParamsGlobals["list_post_dur"] = list_post_dur

        # SANITY CHECK
        for k, v in self.ParamsGlobals.items():
            assert v is not None

        if PRINT:
            print("Updated self.ParamsGlobals:")
            for k, v in self.ParamsGlobals.items():
                print(k, ' = ' , v)

    def extract_snippets_strokes(self, DS, sites_keep, pre_dur, post_dur,
            features_to_get_extra=None):
        """ Extract popanal, one data pt per stroke
        """
        assert False, "use snippets_extract_bystroke"

        # 2) Extract snippets
        ListPA = self.SN.popanal_generate_alldata_bystroke(DS, sites_keep, align_to_stroke=True, 
                                                      align_to_alternative=[], 
                                                      pre_dur=pre_dur, post_dur=post_dur,
                                                      use_combined_region=False,
                                                      features_to_get_extra=features_to_get_extra)
        assert len(ListPA)==1
        return ListPA

    def extract_snippets_trials(self, trials, sites_keep, list_events, 
            list_pre_dur, list_post_dur, list_features_extraction):
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
            levels = sort_mixed_type(data[var].unique().tolist())
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
                levels = sort_mixed_type(self.DfScalar[feature].unique().tolist())
                self.Params["map_var_to_levels"][feature] = levels


    ######################################## working with sessions
    def _session_extract_all(self):
        """
        RETURNS:
        - list of sessions.
        """

        if self.SNmult is None:
            # then is a single 
            assert self._CONCATTED_SNIPPETS==False
            return [self.SN]
        else:
            # Then is a concatted.
            assert self._CONCATTED_SNIPPETS
            return self.SNmult.SessionsList

    def _session_extract_sn_and_trial(self, ind_df=None):
        """ Given row of self.DfScalar, extract the SN and trial within the 
        SN. Useful if this is a concatted across mult session.
        PARAMS:
        - ind_df, index in to self.DfScalar. if None, then takes the first index,
        You should use this onlyu for generic things.
        """

        if ind_df is None:
            ind_df = 0

        trial_neural = self.DfScalar.iloc[ind_df]["trial_neural"]

        if self.SNmult is None:
            # then is a single 
            sn = self.SN
        else:
            # Then is a concatted.
            assert self._CONCATTED_SNIPPETS
            sn_idx = self.DfScalar.iloc[ind_df]["session_idx"]
            sn_neural = self.DfScalar.iloc[ind_df]["session_neural"]
            assert sn_idx==sn_neural, "sn_idx and sn_neural differ..."
            sn = self.SNmult.SessionsList[sn_idx]

            # sanity check
            tc1 = self.DfScalar.iloc[ind_df]["trialcode"]
            tc2 = sn.datasetbeh_trial_to_trialcode(trial_neural)
            assert tc1 == tc2

        return sn, trial_neural


    # def session_sitegetter_map_site_to_region(self, chan):
    #     sn, _ = self._session_extract_sn_and_trial()
    #     return sn.sitegetter_map_site_to_region(chan)

    def session_sitegetter_summarytext(self, chan):
        sn, _ = self._session_extract_sn_and_trial()
        return sn.sitegetter_summarytext(chan)

    # def session_plot_raster_create_figure_blank(self, duration, n_raster_lines, 
    #         n_subplot_rows=1, nsubplot_cols=1, 
    #         reduce_height_for_sm_fr=False, sharex=True):
    #     sn, _ = self._session_extract_sn_and_trial()
    #     return sn._plot_raster_create_figure_blank(self, duration, 
    #         n_raster_lines, n_subplot_rows,
    #         nsubplot_cols, reduce_height_for_sm_fr, sharex)

    # def session_plot_raster_line_mult(self, ax, list_spiketimes, alignto_time=0., 
    #     raster_linelengths=0.9, alpha_raster = 0.9, 
    #     xmin = None, xmax = None, ylabel_trials=None):
    #     sn, _ = self._session_extract_sn_and_trial()


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
            pa = pa.slice_by_labels("trials", var, [lev])

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

    def datamod_append_outliers(self, return_copy=False):
        """ If you removed outliers, add them back.
        (REversible, beucase outlier is a column)
        RETURNS:
            - modifies self.DfScalar (if return_copy is False)
            - or returns copy, wihtout mod
        """

        cols_pre = self.DfScalar.columns
        if hasattr(self, "DfScalar_OutlierRows"):
            if self.DfScalar_OutlierRows is not None:
                if return_copy:
                    DfScalar = self.DfScalar.copy()
                    DfScalar = pd.concat([DfScalar, self.DfScalar_OutlierRows]).reset_index(drop=True)
                    try:
                        if not cols_pre.tolist()==DfScalar.columns.tolist():
                            print(cols_pre)
                            print(DfScalar)
                            assert False
                        return DfScalar
                    except ValueError as err:
                        print("********************")
                        print(cols_pre)
                        print(DfScalar.columns)
                        print(type(cols_pre[0]))
                        print(type(DfScalar.columns[0]))
                        from pythonlib.tools.checktools import check_objects_identical
                        check_objects_identical(cols_pre.tolist(), DfScalar.columns.tolist())
                        raise err
                else:
                    # Mutate
                    self.DfScalar = pd.concat([self.DfScalar, self.DfScalar_OutlierRows]).reset_index(drop=True)
                    # and delete this, so you don't retry this
                    self.DfScalar_OutlierRows = None
                    if not cols_pre.tolist()==self.DfScalar.columns.tolist():
                        print(cols_pre)
                        print(self.DfScalar)
                        assert False

    def datamod_remove_outliers(self):
        """ Remove outliers based on fr_scalar, only for high fr outliers,
        defualt is 3.5x std + mean.
        RETURNS:
        - modifies self.DfScalar, removing outliers, which are rows saved in
        self.DfScalar_OutlierRows
        NOTE: will do diff thing every time run.
        """
        from pythonlib.tools.pandastools import aggregThenReassignToNewColumn, applyFunctionToAllRows

        if hasattr(self, "DfScalar_OutlierRows"):
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

        ## Save just the rows that are removed as outliers. saves space
        self.DfScalar_OutlierRows = df[df["outlier_remove"]==True]

        # 3) remove the outliers
        # self.DfScalar_BeforeRemoveOutlier = df.copy()

        df = df[df["outlier_remove"]==False].reset_index(drop=True)
        self.DfScalar = df

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
    # def dataextract_as_popanal_good(self, event, pre_dur, post_dur):
    #     """
    #     """

    #     # Prune dfthis to the desired temporal window
    #     pre_dur = self.ParamsGlobals["PRE_DUR_CALC"]
    #     post_dur = self.ParamsGlobals["POST_DUR_CALC"]
    #     # fr_which_version = self.Params["fr_which_version"]
    #     dfthis = self.datamod_prune_time_window(dfthis, pre_dur, post_dur)


    def dataextract_as_popanal_good(self, DF, var_trial="index_datapt",
                                    which_fr_sm = "fr_sm_sqrt", chans_needed=None,
                                    balance_chans_trials=False, list_features_extraction=None,
                                    max_frac_trials_lose=None):
        """ Convert this slice of self.DfScalar to popanal, across chans, trials, and time.
        Makes sure that this is clean (i.e, unique and completely balanced balues for
        chans and trials), and this is what you usually want for population analyses. Doesnt
        care about balancing varible features, that you can do later.
        If a var_trial level (e.g., trial) doesnt ahve all chans, then will throw that trial out
        entirely.
        PARAMS:
        - DF, slice of self.DfScalar, this must have only one row for each (var_trial, chan).
        This means that it can't have multiple events. i.e. this is a single event.
        - var_trial, the column holding the trial variable. 
        - chans_needed, list of ints, to pick out exactly these, in this order, for each trial.
        if None, then uses what find for first trial, and throws error if cannot extract 
        these from each trial.
        - balance_chans_trials, bool, if True, then helps ensure that each conjunction of
        trial x chan has at least one datapt (fails otherwise).
        - max_frac_trials_lose, either None(Ignore) or fraction, in which case will throw exception
        if lose more trials than max_frac_trials_lose
        """
        from pythonlib.tools.pandastools import slice_by_row_label
        from pythonlib.tools.pandastools import conjunction_vars_prune_to_balance

        self.datamod_append_unique_indexdatapt()

        # print(self.DfScalar[var_trial])
        # assert False
        if balance_chans_trials:
            # preferably drop trials...
            DF, _ = conjunction_vars_prune_to_balance(DF, "chan", var_trial, 
                                                               PLOT=False, prefer_to_drop_which=2)

        # Times, uses the first index.
        assert self._sanity_fr_sm_times_identical(), "otherwise they have diff times..."
        times = DF.iloc[0]["fr_sm_times"][0,:].tolist()

        # Trials, get sorted list
        list_idx = sorted(DF[var_trial].unique().tolist())

        # Chans, will collect as go.
        list_chansthis = []
        if chans_needed is not None:
            # check that its unique
            assert sorted(list(set(chans_needed))) == sorted(chans_needed)

        if list_features_extraction is None:
            list_features_extraction = []
        list_features_extraction = list_features_extraction + self.Params["list_features_extraction"] + ["trialcode"] + [var_trial]
        list_features_extraction = list(set(list_features_extraction))

        ######### COLLECT DATA across all trials
        list_frmat = []
        list_idx_kept =[]
        out_features = []
        ct = 0
        for idx in list_idx:
            dfthis = DF[(DF[var_trial]==idx)] # len num sites
            if not len(dfthis)==len(chans_needed):
                print("------------", idx)
                print(len(chans_needed))
                print(chans_needed)
                print(len(dfthis))
                print(dfthis["chan"].tolist())
                assert False, "probably because removed outliers"

            # if len(dfthis)>1:
            #     print(len(dfthis))
            #     print(dfthis["event"])
            #     print(dfthis["trialcode"])
            #     print(dfthis[var_trial])
            #     assert False

            # try to slice the desired chans
            if chans_needed is None:
                # its the first trial. use its chans
                assert idx == list_idx[0]
                chans_needed = sorted(dfthis["chan"].tolist())

                # check that its monotic increasing
                # print(chans_needed)
                # print(list(set(chans_needed)))
                assert sorted(list(set(chans_needed))) == sorted(chans_needed)
        
            # use the inputed chans
            # print(dfthis["chan"].value_counts)
            try:
                dfthis = slice_by_row_label(dfthis, "chan", chans_needed, reset_index=True,
                                            assert_exactly_one_each=True)
            except NotEnoughDataException:
                # Skip this datapt, it doesnt have all chans...
                print(f"BAD (this datpt {idx} doesnt have data across all these chans):", dfthis["chan"].unique())
                continue
            # print("GOOD:", dfthis["chan"].unique())
            # print(idx)
            # if not chansthis == chans_needed:
            #     print(f"Skipping datapt {idx} , since it has too few chans: {len(chansthis)}/{len(chans_needed)}")
            #     continue
            list_chansthis.append(dfthis["chan"].tolist())
            list_idx_kept.append(idx)
            
            # extract frmat
            frmat = np.stack(dfthis[which_fr_sm].tolist())
            list_frmat.append(frmat)
            
            # Extract features for this trial.
            tmp ={}
            nchecked = 0
            for feat in list_features_extraction:
                # rows should be diff sites, so they should have identical
                # features...
                try:
                    if not len(dfthis[feat].unique())==1:
                        print(dfthis[feat].value_counts())
                        print("Unique:", dfthis[feat].unique())
                        print(len(dfthis))
                        print(idx, feat)
                        assert False, "each datapt index should have one value for thisf eature (since its one datapt).."
                    nchecked+=1
                except TypeError as err:
                    # ignore.... (if it succeeds for eveyrthing above, unlikely below).
                    pass
                    # first = dfthis[feat][0]
                    # for item in dfthis[feat]:
                    #     assert item == first
                value = dfthis[feat][0]
                tmp[feat] = value
            assert nchecked/len(list_features_extraction)>0.5, "check at least half of vartiables..."
            out_features.append(tmp)
            ct+=1
        print(f"Colected {ct} out of {len(list_idx)} datapts.")
        print("NOTE: missed datapts are likely because of removed outliers")

        if max_frac_trials_lose is not None:
            frac_lost = 1-ct/len(list_idx)
            if frac_lost>max_frac_trials_lose:
                print("Not enough data, likely you DfScalar excludes outlier chan x trials. You should concat outlier rows back to dfscalar")
                raise NotEnoughDataException

        # import pandas as pd
        df_features = pd.DataFrame(out_features)

        # check that each frmat is (sites, 1, times), with identical sites.
        chans_prev = None
        for chans in list_chansthis:
            if chans_prev is None:
                chans_prev = chans
                continue
            if not chans==chans_prev:
                print(chans)
                print(chans_prev)
                assert False
            chans_prev = chans

        assert len(list_frmat)>0, "no data!!!"

        # Construct a PA 
        frate_mat = np.concatenate(list_frmat, axis=1)
        # sn = SP.SNmult.SessionsList[0]
        sn, _ = self._session_extract_sn_and_trial(0)    
        PA = sn._popanal_generate_from_raw(frate_mat, times, chans_needed,
                                           df_features, list_features_extraction)

        return PA

    def _dataextract_as_popanal_singlesite_OLD(self, dfthis, sites, time_take_first=True,
                                               list_cols=None):
        """ Low-level to convert dfthis to popanal. Should
        be made obsolete, as this doesnt really do much.
        """
        assert len(dfthis)>0
        assert len(sites)==1, "not codede yet for more sites..."

        # Extract frmat
        frate_mat = np.stack(dfthis["fr_sm"].tolist())
            
        # get time
        if time_take_first:
            times = dfthis["fr_sm_times"].tolist()[0].squeeze()
        else:
            assert False, "check all unqiue? take each one?"

        # Reshape so is (chan=1, ndat, ntime)
        nchan = frate_mat.shape[1]
        ntime = frate_mat.shape[2]
        ndat = frate_mat.shape[0]
        assert nchan==1
        assert ntime==len(times)
        frate_mat = frate_mat.reshape((nchan, ndat, ntime))

        if list_cols is None:
            list_cols = self.Params["list_features_extraction"] + ["index"] + ["event"]
            list_cols = list(set(list_cols))

        # generate popanal
        sn, _ = self._session_extract_sn_and_trial()
        PA = sn._popanal_generate_from_raw(frate_mat, times, sites, dfthis, list_cols)

        return PA

    def _dataextract_as_popanal_conjunction_vars_OLD(self, var, vars_others=None, site=None,
                                                     event=None, list_cols=None,
                                                     OVERWRITE_n_min=None, OVERWRITE_lenient_n=None,
                                                     balance_same_levels_across_ovar=False
                                                     ):
        """ [OLD] extract dict of PAs, one for each level of vars_others,
        where the data within the PA has all levels of var.
        """

        assert site is not None, "not coded yet. assumes this for below [site]"
        _, dict_lev_df, levels_var = self.dataextract_as_df_conjunction_vars(var, vars_others, site, event,
                                                                             OVERWRITE_n_min=OVERWRITE_n_min, OVERWRITE_lenient_n=OVERWRITE_lenient_n,
                                                                             balance_same_levels_across_ovar=balance_same_levels_across_ovar)


        if list_cols is None:
            if vars_others is not None:
                list_cols = self.Params["list_features_extraction"] + ["index"] + [var] + vars_others + ["event"]
            else:
                list_cols = self.Params["list_features_extraction"] + ["index"] + [var] + ["event"]
            list_cols = list(set(list_cols))

        dict_lev_pa = {}
        for lev_other, dfthis in dict_lev_df.items():
            pa = self._dataextract_as_popanal_singlesite_OLD(dfthis, [site], list_cols=list_cols)
            dict_lev_pa[lev_other] = pa

        return dict_lev_pa, levels_var 

    def dataextract_as_df_good(self, chan=None, event_aligned=None, var=None, var_level=None,
                               dfthis=None, list_chan=None, pre_dur=None, post_dur=None):
        """ 
        Extract dfthis, slice of self.DfScalar
        PARAMS
        - chan, int
        - event, unique event (00_..) into event_aligned
        - var, var_level, either both None (ignore var), or string and value.
        - list_chan, list of ints, gets these chans. Only this or chan can be not None
        - pre_dur, post_dur, times (sec) relative to 0 (alingment) to prune sm fr.
        RETURNS:
        - df, a copy, and index reset
        """

        if list_chan is not None and chan is not None:
            print(list_chan)
            print(chan)
            assert False, "cannot use both..."
        
        assert (var_level==None) == (var==None)
            
        if dfthis is None:
            dfthis = self.DfScalar

        if chan is not None:
            dfthis = dfthis[(dfthis["chan"]==chan)]
            assert len(dfthis)>0

        if list_chan is not None:
            dfthis = dfthis[(dfthis["chan"].isin(list_chan))]
            assert len(dfthis)>0

        if event_aligned is not None:
            dfthis = dfthis[(dfthis["event_aligned"] == event_aligned)]
            assert len(dfthis)>0

        if var is not None:
            dfthis = dfthis[(dfthis[var]==var_level)]
            assert len(dfthis)>0

        dfthis = dfthis.copy().reset_index(drop=True)

        # prune fr?
        if pre_dur is not None:
            dfthis = self.datamod_prune_time_window(dfthis, pre_dur, post_dur)

        return dfthis

    # def dataextract_as_df_conjunction_vars_multsites(self, ...,
    #     do_balance=False, balance_var=None, list_balance_vars_others=None):


    def dataextract_as_df_multsites_wrapper(self, sites, event_aligned,
                                            var=None, list_vars_others=None, do_balance=False,
                                            pre_dur=None, post_dur=None, trial_var = "index_datapt",
                                            PRINT = False,
                                            exclude_othervar_levels_missing_any_var_level=False):
        """
        GOOD, a single method to extract structured data,
        allowing for multiple sites, structured by split by vars, etc, ensuring
        you have balanced data, and pruning otherwise.
        PARAMS:
        - sites, list of ints
        - event_aligned, str, e.b., "00_..."
        - var, if not None, then will prune dataset to be sure you have enough
        data for each level of var.
        - list_vars_others, list of str, if not None, then looks at conjunctions
        of levels of var and vars_others for pruning.
        - do_balance, if True, then ensures that each level of var has at least one data
        for each level of list_var_others
        NOTES:
        Option 1: ignore variables:
            DEFAULT
        Option 2: get conjunction variables, but dont care about balance
            input var and list_vars_others
        Option 3: also balance
            also input do_balance=True
        """

        # Always start with geting raw data, since this function allows getting multiple sites, ignoring variables
        # print(event_aligned)
        # print(self.DfScalar["event"])
        dfthis = self.dataextract_as_df_good(event_aligned=event_aligned, list_chan=sites, pre_dur=pre_dur, post_dur=post_dur)

        # Get conjunction vars, and balance the data?
        if var:
            # This is a bit tricky, since you have multiple sites. So sample size check (for deciding which 
            # levels with neough data) will be inflated data. So take the first site. get its
            # conjunctions, then use the extracted datapts to extract across all sites.
            
            # NOTE: This assumes that each site has the same datapts... which is usually the case.
            self._sanity_trial_and_chans_are_balanced(dfthis)
            sites = dfthis["chan"].unique().tolist()
            
            # 2) Extract, pruning by sample sizes of conjunctions.
            # just pick the first site, since now all sites have exacly the same trials.
            _dfthis_single, _, _ = self.dataextract_as_df_conjunction_vars(var, list_vars_others, 
                                            site=sites[0], DFTHIS=dfthis, 
                                            DEBUG_CONJUNCTIONS=False, balance_no_missed_conjunctions=do_balance,
                                            exclude_othervar_levels_missing_any_var_level=exclude_othervar_levels_missing_any_var_level)

            if len(_dfthis_single)>0:
                #### SECOND, use result from single site to filter all data
                # dfthis tells you which datinds to keep to still have good n data
                # Apply this filter to keep only these datinds for all sites.
                n1 = len(dfthis)
                list_index_datapt = _dfthis_single[trial_var].unique().tolist()
                dfthis = dfthis[dfthis[trial_var].isin(list_index_datapt)].reset_index(drop=True)
                n2 = len(dfthis)
                if PRINT:
                    print("Starting len:", n1)
                    print("These are the final good index datapts: ")
                    print(list_index_datapt)
                    print("Ending len:", n2)
            else:
                # then return empty dataframe, since you dont keep any datpts.
                return pd.DataFrame([])
        else:
            assert do_balance==False, "must pass in var..."   
        
        return dfthis

    def dataextract_as_df_split_into_var_conjunctionvar(self, dfdat, var, 
            vars_others=None, levels_var = None, levels_othervar=None):
        """
        Split dataframe (wiuthout modifuying any data or removing any rows) in
        multiuple flexible ways.
        PARAMS:
        - var, column, will split dat by its levels
        - vars_others, list of str columns, will first make new columns "vars_others" grouping this,
        then split data by conjucntion of var and "vars_others"
        RETURNS:
        - DICT_DF_DAT, keys are levels or conjhunctions of levels (depending on whether
        vars_others is None) and df. Sum of lenghts of dfs over all values equals len dfdat
        - levels_var, levels_othervar, list of levels, or None for levels_othervar if vars_others is None
        NOTE: DICT_DF_DAT might not have keys for each conjunction of levels of var and levels_othervar...
        """
        from pythonlib.tools.pandastools import extract_with_levels_of_var, append_col_with_grp_index

        if levels_var:
            assert isinstance(levels_var, list)
        if levels_othervar:
            assert isinstance(levels_othervar, list)

        # print(var, levels_var)
        # print(vars_others, levels_othervar)
        # assert False
        if vars_others is not None:
            # 1. combine othervars into a single var
            dfdat = append_col_with_grp_index(dfdat, vars_others, "vars_others", use_strings=False)

            if levels_var is not None:
                dfdat = dfdat[dfdat[var].isin(levels_var)]
            if levels_othervar is not None:
                dfdat = dfdat[dfdat["vars_others"].isin(levels_othervar)]

            # 2. take conjunction of var and othervars
            dfdat = append_col_with_grp_index(dfdat, [var, "vars_others"], "dummy", use_strings=False)
            DICT_DF_DAT, _levels = extract_with_levels_of_var(dfdat, "dummy")
            levels_var = sort_mixed_type(set([x[0] for x in _levels]))
            levels_othervar = sort_mixed_type(set([x[1] for x in _levels]))

            # for lev in levels_var:
            #     for levother in levels_othervar:
            #         key = (lev, levother)
            #         if key not in DICT_DF_DAT.keys():
            #             print(1, DICT_DF_DAT)
            #             print(2, DICT_DF_DAT.keys())
            #             print(3, levels_var)
            #             print(4, levels_othervar)
            #             print(5, key)
            #             print(6, var)
            #             print(7, vars_others)
            #             print(8, dfdat[var].unique())
            #             print(9, dfdat["vars_others"].unique())
            #             print(10, dfdat["dummy"].unique())
            #             assert False, "bug... (fix it)."

        else:
            DICT_DF_DAT, levels_var = extract_with_levels_of_var(dfdat, var, levels_var)
            levels_othervar = None

        assert sum([len(df) for df in DICT_DF_DAT.values()]) == len(dfdat)
        # # Optionally, split data based on levels of var and vars_others
        # DICT_DF_DAT = {}
        # if vars_others is not None:
        #     for levvar, _df in dict_var_df.items():
        #         dict_var_df_this, levels_var_this = extract_with_levels_of_var(_df, "vars_others")
        #         for levothervar, _df in dict_var_df_this.items():
        #             key = (levothervar, levvar)
        #             DICT_DF_DAT[key] = _df
        # else:
        #     for levvar, _df in dict_var_df.items():
        #         key = (levvar, )
        #         DICT_DF_DAT[key] = _df

        return DICT_DF_DAT, levels_var, levels_othervar


        #     dict_var_othervar_df = {}

        #     assert False, "make sure dataextract_as_df_conjunction_vars is efficient if vars are 0."
        #     _, dict_othervar_df, _ = self.dataextract_as_df_conjunction_vars(var, vars_others,
        #         DFTHIS=dfdat, OVERWRITE_n_min=0, OVERWRITE_lenient_n=0)
        #     for levother, df in dict_othervar_df.items():
        #         dict_var_df, _ = extract_with_levels_of_var(df, var, levels=levels_var)

        #         for lev, _df in dict_var_df.items():
        #             key = (levother, lev)
        #             dict_var_othervar_df[key] = _df

        # for k, v in dict_var_othervar_df.items():
        #     print(k, len(v))

        # # Make a dict holding each df that you want to plot a different color
        # DICT_DF_DAT = dict_lev_df

        # print("each df to plot (diff color):")
        # for k, v in DICT_DF_DAT.items():
        #     print(k, len(v))


    def dataextract_as_df_conjunction_vars(self, var, vars_others=None, site=None,
        event=None, DEBUG_CONJUNCTIONS=False,
        OVERWRITE_n_min=None, OVERWRITE_lenient_n=None,
        DFTHIS = None, balance_no_missed_conjunctions=False,
        exclude_othervar_levels_missing_any_var_level=False,
        PRINT_AND_SAVE_TO=None,
        ignore_values_called_ignore=True,
           balance_same_levels_across_ovar=False):
        """ Helper to extract dataframe (i) appending a new column
        with ocnjucntions of desired vars, and (ii) keeping only 
        levels of this vars (vars_others) that has at least n trials for 
        each level of a var of interest (var).
        Useful becuase the output is ensured to have all levels of var, and (wiht 
        some mods) could be used to help balance the dataset.
        NOTE: can choose to allow levels of vars_others to have only a subset of
        levels of var, by makign the param lenient_allow_data_if_has_n_levels an int.
        PARAMS:
        - var, str, the variabiel you wisht ot use this dataset to compute 
        moudlation for.
        - vars_others, list of str, variables that conjucntion will define a 
        new goruping var.
        - site, either None (all data) or int.
        - n_min, min n trials desired for each level of var. will only keep
        conucntions of (vars_others) which have at least this many for each evel of
        var.
        - balance_no_missed_conjunctions, bool, if True, then makes sure the resulting
        dfthis is "square" in that each level of var has at least some data for
        each level of vars_others. Does this in an interative fashion (see inner code).
        - exclude_othervar_levels_missing_any_var_level, bool, if True, then only keeps
        levels of othervar which have at least one datapt for each level of var.
        EG:
        - you wish to ask about shape moudlation for each comsbaiton of location and 
        size. then var = shape and vars_others = [location, size]
        RETURNS:
        dfthis, dict_lev_df, levels_var
        - dataframe, with new column "vars_others"
        - dict, lev:df
        - levels_var, list of str, levels of var.
        """

        assert site in self.DfScalar["chan"].unique().tolist()

        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        import numpy as np

        if balance_no_missed_conjunctions:
            assert vars_others is not None, "need this, or else not sure what to balance.."

        if isinstance(vars_others, tuple):
            vars_others = list(vars_others)

        if DFTHIS is None:
            DFTHIS = self.DfScalar

        if site is not None:
            assert site in DFTHIS["chan"].tolist(), "This channel doesnt exist!!"
        
        if event is not None:
            if event in DFTHIS["event"].tolist():
                event_col = "event"
            elif event in DFTHIS["event_aligned"].tolist():
                event_col = "event_aligned"
            else:
                print(event)
                print(DFTHIS)
                assert False, "This event doesnt exist!!"

        # 1) extract for this site.
        if site is None and event is None:
            dfthis = DFTHIS
        elif event is None:
            dfthis = DFTHIS[(DFTHIS["chan"] == site)]
        elif site is None:
            dfthis = DFTHIS[(DFTHIS[event_col] == event)]
        else:
            dfthis = DFTHIS[(DFTHIS["chan"] == site) & (DFTHIS[event_col] == event)]
        dfthis = dfthis.reset_index(drop=True)

        # Use all unique levels in entire dataset, across all sublevels.
        levels_var = sort_mixed_type(DFTHIS[var].unique().tolist())

        n_min = self.ParamsGlobals["n_min_trials_per_level"]
        lenient_allow_data_if_has_n_levels = self.ParamsGlobals["lenient_allow_data_if_has_n_levels"]

        # Remove rows that have None or na for any of the variables
        if vars_others is None:
            list_vars_check = [var]
        else:
            list_vars_check = vars_others+[var]
        for varthis in list_vars_check:

            # for each variable, remove rows that have nan or None
            tmp = dfthis[varthis].isna()
            # indsdrop = np.where([x is None for x in tmp])
            indsdrop = np.where(tmp)[0]
            if DEBUG_CONJUNCTIONS:
                print(f"var {varthis}, Removing this many rows with None or nan: {sum(indsdrop)}")
            try:
                dfthis = dfthis.drop(indsdrop, axis=0).reset_index(drop=True)
            except Exception as err:
                print(indsdrop)
                print(type(indsdrop))
                print(indsdrop[0])
                print(len(dfthis))
                print(dfthis.index)
                raise err

        # 2) extract_with_levels_of_conjunction_vars
        if False: # is fine to not have, since now I am not constraining that events must exist for all
            # trials (at equal proportion across events)
            assert len(dfthis)>0, "why is empty? incorrect event or sites?"

        # print("HERERER:", len(dfthis))
        if OVERWRITE_n_min:
            n_min = OVERWRITE_n_min
        if OVERWRITE_lenient_n:
            lenient_allow_data_if_has_n_levels = OVERWRITE_lenient_n

        # This overwrites everything:
        if exclude_othervar_levels_missing_any_var_level:
            lenient_allow_data_if_has_n_levels = None

        # print("n_min, lenient_allow_data_if_has_n_levels:", n_min, lenient_allow_data_if_has_n_levels)
        if balance_same_levels_across_ovar:
            # Then prune levels of var, trying to keep all the ovar if possible.
            balance_no_missed_conjunctions = True
            balance_force_to_drop_which = 1
        else:
            balance_force_to_drop_which = None

        dfthis, dict_lev_df = extract_with_levels_of_conjunction_vars(dfthis, var, vars_others, levels_var, n_min,
                                                                      lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels,
                                                                      DEBUG=DEBUG_CONJUNCTIONS,
                                                                      balance_no_missed_conjunctions=balance_no_missed_conjunctions,
                                                                      PRINT_AND_SAVE_TO=PRINT_AND_SAVE_TO,
                                                                      ignore_values_called_ignore=ignore_values_called_ignore,
                                                                      balance_force_to_drop_which=balance_force_to_drop_which,
                                                                      # plot_counts_heatmap_savepath="/tmp/test.png"
                                                                      )
        # print(dfthis[var].unique())
        # print(n_min, lenient_allow_data_if_has_n_levels, balance_no_missed_conjunctions, balance_force_to_drop_which)
        # print(site, event, len(dfthis))
        # for k, dfthis in dict_lev_df.items():
        #     print(k, dfthis[var].unique())
        # assert False

        return dfthis, dict_lev_df, levels_var


    def dataextract_as_df_OLD(self, grouping_variables, grouped_var_col_name):
        """
        OBSOLETE, not used.
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        dfthis = append_col_with_grp_index(self.DfScalar, grouping_variables, 
            grouped_var_col_name, strings_compact=True)

        columns_keep = ["chan", "trialcode", "fr_sm"]
        return dfthis.loc[:, columns_keep + [grouped_var_col_name]]

    def dataextract_as_frmat(self, chan, event=None, var=None, var_level=None, 
        fr_ver="fr_sm", return_as_zscore=False):
        """ 
        [GOOD]
        Extract frmat from self.DfScalar, stacking all instances of this event, and
        (optionally) only this level for this var.
        PARAMS
        - chan, int
        - event, unique event (00_..) into event_aligned
        - var, var_level, either both None (ignore var), or string and value.
        RETURNS:
        - frmat, (ntrials, ntime)
        - times, (ntimes, )
        - dict, holding useful dat for formatitng plots.
        """
        
        from neuralmonkey.utils.frmat import dfthis_to_frmat

        assert (var_level==None) == (var==None)
        
        dfthis = self.dataextract_as_df_good(chan, event, var, var_level)
        # if event is None and var is None:
        #     dfthis = self.DfScalar[(self.DfScalar["chan"]==chan)]
        # elif event is None:
        #     dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar[var]==var_level)]   
        # elif var is None:
        #     dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar["event_aligned"]==event)]    
        # else:
        #     dfthis = self.DfScalar[(self.DfScalar["chan"]==chan) & (self.DfScalar["event_aligned"]==event) & (self.DfScalar[var]==var_level)]   

        frmat, times = dfthis_to_frmat(dfthis, fr_ver=fr_ver, return_as_zscore=return_as_zscore)    

        # Also return useful things for plotting
        ind_aligned_to_event = np.argmin(np.abs(times))
        xticks = [0, ind_aligned_to_event, len(times)-1]
        xtick_labels = [f"{times[i]:.3f}" for i in xticks]
        dict_plot_vals = {
            "ind_aligned_to_event":ind_aligned_to_event,
            "xticks":xticks,
            "xtick_labels":xtick_labels
        }
        return frmat, times, dict_plot_vals


    def _dataextract_as_metrics_scalar(self, dfthis, var=None):
        """ Pass in dfthis directly, to generate an MS object.
        """
        from neuralmonkey.metrics.scalar import MetricsScalar
        
        assert len(dfthis)>0

        # Prune dfthis to the desired temporal window
        pre_dur = self.ParamsGlobals["PRE_DUR_CALC"]
        post_dur = self.ParamsGlobals["POST_DUR_CALC"]
        # fr_which_version = self.Params["fr_which_version"]
        dfthis = self.datamod_prune_time_window(dfthis, pre_dur, post_dur)  
        dfthis = self.datamod_compute_fr_scalar(dfthis, pre_dur, post_dur)

        # Input to Metrics
        # (use this, instead of auto, to ensure common values across all chans)
        list_var = list(set(self.Params["list_features_get_conjunction"] + self.Params["list_features_extraction"]))
        if var is not None:
            list_var.append(var)
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

        dfthis = self.DfScalar[self.DfScalar["chan"]==chan].reset_index(drop=True)
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

    def datamod_prune_time_window(self, df, pre_dur, post_dur):
        """ Return df wehre the following data columns are restricted to
        within pre_dur (engative) and psot_dur: 
            spike_times, fr_sm_times, fr_sm, fr_sm_sqrt
        Also removes these, since they are wrong:
            any coilumn "fr_scalar_*"
        NOTE:
        - takes about 18ms for len 400 df.
        """

        assert len(df)>0
        assert pre_dur < post_dur, "pre_dur usually negative."
        assert pre_dur is not None
        assert post_dur is not None

        def F(x):
            inds = (x["spike_times"]>=pre_dur) & (x["spike_times"]<=post_dur)
            return x["spike_times"][inds]
        dfthis = applyFunctionToAllRows(df, F, "spike_times")

        list_fr_sm_keys = ["fr_sm", "fr_sm_sqrt", "fr_sm_times"]
        assert list_fr_sm_keys[-1] == "fr_sm_times", "this must go last, since it is needed for the earlier ones."

        for fr_sm_key in list_fr_sm_keys:
            if fr_sm_key in dfthis.columns:
                def F(x):
                    inds = (x["fr_sm_times"]>=pre_dur) & (x["fr_sm_times"]<=post_dur)
                    return x[fr_sm_key][inds][None, :] # to get (1, n) instead of (n,)
                dfthis = applyFunctionToAllRows(dfthis, F, fr_sm_key)

        # def F(x):
        #     inds = (x["fr_sm_times"]>=pre_dur) & (x["fr_sm_times"]<=post_dur)
        #     return x["fr_sm"][inds][None, :] # to get (1, n) instead of (n,)
        # dfthis = applyFunctionToAllRows(dfthis, F, "fr_sm")
        
        # if "fr_sm_sqrt" in dfthis.columns:
        #     def F(x):
        #         inds = (x["fr_sm_times"]>=pre_dur) & (x["fr_sm_times"]<=post_dur)
        #         return x["fr_sm_sqrt"][inds][None, :]
        #     dfthis = applyFunctionToAllRows(dfthis, F, "fr_sm_sqrt")

        # ### FINALLY, prune times.
        # def F(x):
        #     inds = (x["fr_sm_times"]>=pre_dur) & (x["fr_sm_times"]<=post_dur)
        #     return x["fr_sm_times"][inds][None, :]
        # dfthis = applyFunctionToAllRows(dfthis, F, "fr_sm_times")

        # Remove any scalars
        cols = [c for c in dfthis.columns if "fr_scalar_" in c]
        dfthis = dfthis.drop(cols, axis=1)

        # sanity check
        cols = [c for c in dfthis.columns if "fr_sm_" in c]
        assert all([c in ["fr_sm", "fr_sm_sqrt", "fr_sm_times"] for c in cols])

        return dfthis

    def datamod_compute_fr_scalar(self, df, pre_dur=None, post_dur=None, 
        fr_which_version=None):
        """
        Appends columns with fr scalar values, given a time window of interest
        PARAMS:
        - pre_dur, post_dur, if None, then uses those values from 
        initial extraction in self.Params
        RETURNS:
        - df copy, with new columns fr_scalar_raw, fr_scalar, where the
        latter is either raw or sqrt, based on fr_which_version
        """

        # print("Recomputing fr scalar using (pre_dur, post_dur, fr_which_version)", (pre_dur, post_dur, fr_which_version))

        if pre_dur is None:
            pre_dur = self.Params["list_pre_dur"][0]
        if post_dur is None:
            post_dur = self.Params["list_post_dur"][0]
        if fr_which_version is None:
            fr_which_version = self.Params["fr_which_version"]

        # Compute fr scalar
        dur = post_dur - pre_dur
        def F(x):
            inds = (x["spike_times"]>=pre_dur) & (x["spike_times"]<=post_dur)
            nspk = sum(inds)
            rate = nspk/dur
            return rate
        df = applyFunctionToAllRows(df, F, "fr_scalar_raw")

        # tgransform the fr if desired
        if fr_which_version=="raw":
            df["fr_scalar"] = df["fr_scalar_raw"] 
        elif fr_which_version=="sqrt":
            df["fr_scalar"] = df["fr_scalar_raw"]**0.5
        else:
            print(fr_which_version)
            assert False

        # print("Recomputing fr done!")
        return df

    def datamod_append_bregion(self, df):
        """ Appends bregion for each row, using the chan in df["chan"]
        RETURNS:
        - copy of df, with column "bregion"
        """
        sn, _ = self._session_extract_sn_and_trial()
        def F(x):
            region = sn.sitegetterKS_map_site_to_region(x["chan"])
            return region
        return applyFunctionToAllRows(df, F, "bregion")

    def datamod_append_unique_indexdatapt(self):
        """ Assign to each row a unique index correspoding to a "datapoint"
        which dpeends on if it is "trial" (trialcode)  or "stroke" (trialcode x stroke index) 
        level.
        RETURNS:
        - modifies self.DfScalar with new column "index_datapt"
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index

        if "index_datapt" not in self.DfScalar.columns:
            if self.Params["which_level"]=="trial":
                grp = ["trialcode"]
            elif self.Params["which_level"] in ["stroke", "stroke_off", "substroke", "substroke_off"]:
                grp = ["trialcode", "stroke_index"]
            elif self.Params["which_level"] in ["saccade_fix_on", "fixon", "flex"]:
                grp = ["trialcode", "event_idx_within_trial"]
            else:
                assert False

            try:
                for g in grp:
                    if g not in self.DfScalar.columns:
                        print("HERE")
                        print(g)
                        print(self.DfScalar)
                        assert False
            except Exception as err:
                print("HERE")
                print(g)
                print(self.DfScalar)
                raise err

            self.DfScalar = append_col_with_grp_index(self.DfScalar, grp, "index_datapt", use_strings=False)

    # def datamod_balance_conjunction_of_levels(self, dfthis, var1, var2,
    #     balance_prefer_to_drop_which=None, PLOT=False):
    #     """ Balance this dataset so that each level of var has at least one
    #     datapt conjuicntion with each level of othervar. 
    #     """
    #     from pythonlib.tools.pandastools import conjunction_vars_prune_to_balance
    #     dfthisout, dfcounts = conjunction_vars_prune_to_balance(dfthis, var1, var2, 
    #         prefer_to_drop_which=balance_prefer_to_drop_which,
    #         PLOT=DEBUG);


    ################ MODULATION BY VARIABLES
    def _modulationgood_extract_fr(self, which_sm_fr="fr_sm_sqrt",
        pre_dur=None, post_dur=None, fr_var = "fr_scalar",
        dfthis=None):
        """ Compute mean fr and store in self.DfScalar["fr_scalar"], by 
        using smoothed fr (windowed with pre_dur, post_dur)
        PARAMS:
        - which_sm_fr, string, column to use (e.g,, fr_sm_sqrt) 
        - pre_dur, post_dur, if None, then uses what's in ParamsGlobal[PRE_DUR_CALC], etc.
        - dfthis, if None, then uses self.DfScalar
        RETURNS:
        - modifies dfthis["fr_scalar"], and returns it
        """

        if dfthis is None:
            dfthis = self.DfScalar

        if pre_dur is not None or post_dur is not None:
            self.globals_update(PRE_DUR_CALC=pre_dur, POST_DUR_CALC=post_dur)
        pre_dur = self.ParamsGlobals["PRE_DUR_CALC"]
        post_dur = self.ParamsGlobals["POST_DUR_CALC"]

        assert self._sanity_fr_sm_times_identical(), "cannot use a single row's times"
        times = dfthis.iloc[0]["fr_sm_times"]
        inds = (times>=pre_dur) & (times<=post_dur)
        inds = inds.squeeze()

        # concat, slice, and take mean to get sclaar
        frmat = np.concatenate(dfthis[which_sm_fr].tolist(), axis=0)
        frscal = np.mean(frmat[:, inds], axis=1)

        # update
        dfthis[fr_var] = frscal    

        return dfthis


    def modulationgood_compute_fr_quick(self, var, other_var_str="vars_others", fr_var = "fr_scalar",
            dfthis = None):
        """ Compute datafromae hodling scalar fr across (chan, levels)
        PARAMS:
        - var, to get the levels of interst
        - dfthis, if None , then uses elf.DfScalar
        RETURNS:
        - df_fr, each row is a single (chan, event)
        - df_fr_levels, each row is a single (chan, event, level_of_var)
        """
        from pythonlib.tools.pandastools import aggregGeneral

        # fr_which_version = self.Params["fr_which_version"]
        if dfthis is None:
            dfthis = self.DfScalar

        if "fr_scalar" not in dfthis.columns:
            if False:
                # Compute fr based on updated pre and post dur.
                pre_dur = self.ParamsGlobals["PRE_DUR_CALC"]
                post_dur = self.ParamsGlobals["POST_DUR_CALC"]
                print("Computing fr scalar quickly (not windowed), to help prune low Fr neurons...")
                dfthis = self.datamod_compute_fr_scalar(dfthis, pre_dur, post_dur)
                print("Done...")
            else:
                # THe above is too slow. here compute from sm fr.
                self._modulationgood_extract_fr(fr_var=fr_var,
                    dfthis=dfthis)

        df_fr = aggregGeneral(dfthis, group=["chan", "event_aligned"], values = [fr_var])
        # give bregion
        df_fr = self.datamod_append_bregion(df_fr)
        df_fr["event"] = df_fr["event_aligned"]
        df_fr["val"] = df_fr[fr_var]
        df_fr["val_kind"] = fr_var

        df_fr_levels = aggregGeneral(dfthis, group=["chan", "event_aligned", var, other_var_str], values = [fr_var])
        # give bregion
        df_fr_levels = self.datamod_append_bregion(df_fr_levels)
        df_fr_levels["event"] = df_fr_levels["event_aligned"]
        df_fr_levels["val"] = df_fr_levels[fr_var]
        df_fr_levels["val_kind"] = "fr_each_level"
        df_fr_levels["var"] = df_fr_levels[var]

        return df_fr, df_fr_levels

    def modulationgood_plot_WRAPPER(self, df_var, df_fr, df_fr_levels, 
            list_eventwindow_event, var, vars_conjuction, 
            sdir_base, N_WAYS=1, PLOT_EACH_CHAN=False, PLOT_EACH_EVENT=False):
        """
        Overview of modulation by variable, conditioned on other variables, already
        extacted. 
        See its use in 
        """
        ####### SUmmary plot of anova

        assert df_fr is None, "not used"
        assert df_fr_levels is None, "not used"

        sdir = f"{sdir_base}/modulation"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)

        print("** Plotting summarystats")
        self.modulationgood_plot_summarystats(df_var, None, None, savedir=sdir) 
        plt.close("all") 

        ######## 1b) heatmap
        print("** Plotting heatmaps")
        sdir = f"{sdir_base}/modulation_heatmap"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        self.modulationgood_plot_brainschematic(df_var, sdir) 

        ######## if this is two-way anomva, then also plot after adding up main and interactino
        if N_WAYS==2:
            # print("** Plotting heatmaps")
            # sdir = f"{sdir_base}/modulation_2anova"
            # os.makedirs(sdir, exist_ok=True)
            # print(sdir)                
            self.modulationgood_plot_twoway_summary(df_var, sdir_base) 

        ######### Plot moduulation for each chans
        sdir = f"{sdir_base}/each_chan_summary"
        os.makedirs(sdir, exist_ok=True)
        fig = sns.catplot(data=df_var, x="chan", y="val", row="event", hue="bregion", aspect=10, kind="point")
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        fig.savefig(f"{sdir}/allchans_modulation.pdf")
        plt.close("all")

        ######### Plot moduulation for each chans
        if PLOT_EACH_CHAN:
            print("** Plotting modulation each chan")
            sdir = f"{sdir_base}/each_chan"
            os.makedirs(sdir, exist_ok=True)
            self.modulationgood_plot_each_chan(df_var, var, vars_conjuction, sdir)
            plt.close("all")           

        ########### SEPARATE PLOT FOR EACH EVENT
        # events_already_done = []
        for event_window, event in list_eventwindow_event:
            print("** Making plots for this event_window: ")
            print(event_window)

            sdir_base_this = f"{sdir_base}/EACH_EVENT/{event_window}"
            os.makedirs(sdir_base_this, exist_ok=True)
            print(f"Saving this event, {event_window}, to {sdir_base_this}")

            df_var_this = df_var[df_var["event"]==event_window]

            if len(df_var_this)==0:
                print("SKipping this event, since no data... ", event_window)
                print(len(df_var))
                print(df_var["event"].value_counts())
                print(event_window, event)
                # assert False
                continue
                
            ####### SUmmary plot of anova
            if PLOT_EACH_EVENT:
                if get_z_score:
                    sdir = f"{sdir_base_this}/modulation_v2"
                    os.makedirs(sdir, exist_ok=True)
                    print(sdir)

                    print("** Plotting summarystats")
                    self.modulationgood_plot_summarystats_v2(df_var_this, savedir=sdir) 
                    plt.close("all")
                
            ############## PRINT INFO
            spath = f"{sdir_base_this}/variable_conjunctions-actually_used.txt"
            _, dict_levs, levels_var = self.dataextract_as_df_conjunction_vars(var, 
                vars_conjuction, site=self.Sites[0], event=event,
                PRINT_AND_SAVE_TO=spath)

            if False:
                # incorrectly includes all levels. fix if want to keep this
                df = self.DfScalar[(self.DfScalar["chan"] == self.Sites[0])]
                spath = f"{sdir_base_this}/variable_conjunctions-all_possible.txt"
                from pythonlib.tools.pandastools import grouping_print_n_samples
                nmin = self.ParamsGlobals["n_min_trials_per_level"]
                grouping_print_n_samples(df, [var] + vars_conjuction, nmin, spath, True)


    def modulationgood_compute_plot_ALL(self, var, vars_conjuction, 
        score_ver='r2_maxtime_1way_mshuff', SAVEDIR="/tmp",
        PRE_DUR_CALC = None, POST_DUR_CALC= None,
        globals_nmin = None, globals_lenient_allow_data_if_has_n_levels = None,
        FR_THRESH=None, FR_PERCENTILE = 10, 
        list_events = None, list_pre_dur = None, list_post_dur = None,
        DEBUG_CONJUNCTIONS = False,
        PLOT_EACH_CHAN=False,
        get_z_score=False,
        reload_df_var_if_exists=True,
        supervision_keep_only_not_training=True,
        trialcodes_keep=None,
        ANALY_VER = "default",
        params_to_save=None,
        do_only_print_conjunctions=False,
        PLOT_RASTERS = True,
        PLOT_EACH_EVENT=False):
        """
        New version for computing and plotting all modulation (anova) for a given variable. and set
        of conjunction variables. 
        PRE_DUR_CALC = None, POST_DUR_CALC= None, for calculating modulation, differnet from
        what was used for extraction.
        PARAMS;
        - 
            list_events = ["00_fix_touch", "01_samp", "03_first_raise"],
            list_pre_dur = [-0.5, 0.05, -0.1],
            list_post_dur = [-0, 0.6, -0.5],):\
        - DEBUG_CONJUNCTIONS, bool, if true, then for each data extraction prints the n trials
        for each var/othervar conjunctions, to help see what data extracted. 
        - params_to_save, usually the params that hold the meta params. see analy_anova_plot
        - do_only_print_conjunctions, bool, then applies all preprocessing then only prints and
        saves conjunctions of varialbes. allows inspecting this before running entire analysis which
        takes long time.
        """
        from pythonlib.tools.expttools import writeDictToYaml        
        import pickle
        import os
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.pandastools import aggregGeneral
        # var = "chunk_within_rank"
        # vars_conjuction = ['stroke_index', 'gridloc', 'chunk_rank'] # list of str, vars to take conjunction over
        # vars_conjuction = ['gridloc', 'chunk_rank'] # list of str, vars to take conjunction over

        # Helping figure out which variables matter
        assert PRE_DUR_CALC is None, "not used! use list_pre_dur etc"
        assert POST_DUR_CALC is None, "not used! use list_pre_dur etc"

        print(" !!! RUNNING modulationgood_compute_plot_ALL for var and vars_conjuction:")
        print(var)
        print(vars_conjuction)

        assert(np.all(np.diff(self.DfScalar.index)==1)), "do reset index"

        assert isinstance(vars_conjuction, list)

        df_scalar_orig_length = len(self.DfScalar)
        if trialcodes_keep is not None:
            assert isinstance(trialcodes_keep, list)
            # Then prune to just these trialcodes.
            indskeep = self.DfScalar["trialcode"].isin(trialcodes_keep)
            if all(indskeep):
                # do nothing, all are already in "not training"
                pass
            elif all(~indskeep):
                # Lost all trials, this is weird.
                print(trialcodes_keep)
                print(self.DfScalar["trialcode"].tolist())
                assert False
            else:
                # save the old DfScalar
                self.DfScalarBeforeRemoveSuperv = self.DfScalar.copy()
                self.DfScalar = self.DfScalar[indskeep].reset_index(drop=True)
        df_scalar_pruned_length = len(self.DfScalar)
        
        if False:
            # Old version, which automatically pruned supervision stage. Now this is replaced
            # by trialcode_keep
            if supervision_keep_only_not_training:
                # In general, prune supervision (to remove training trials), 
                # unless var or vars_conjuction cares about supervision.
                # if has to prune:
                    # --> stores copy of DfScalar in DfScalarBeforeRemoveSuperv
                    # --> prunes so DfScalar only has no_supervision trials.
                # else:
                    # --> DfScalarBeforeRemoveSuperv = None
                if var=="supervision_stage_concise" or "supervision_stage_concise" in vars_conjuction:
                    # is ok, do not prune supervision stage
                    pass
                else:
                    assert False, "use D.preprocessGood(params=[no_supervision]) instead"
                    indskeep = self.DfScalar["supervision_stage_concise"].isin(LIST_SUPERV_NOT_TRAINING)
                    if all(indskeep):
                        # do nothing, all are already in "not training"
                        pass
                    elif all(~indskeep):
                        # Lost all trials, this is weird.
                        print(LIST_SUPERV_NOT_TRAINING)
                        print(self.DfScalar["supervision_stage_concise"].value_counts())
                        assert False
                    else:
                        # save the old DfScalar
                        self.DfScalarBeforeRemoveSuperv = self.DfScalar.copy()
                        self.DfScalar = self.DfScalar[indskeep].reset_index(drop=True)
            else:
                self.DfScalarBeforeRemoveSuperv = None

        sdir_base = f"{SAVEDIR}/{ANALY_VER}/var_by_varsother/VAR_{var}-OV_{'_'.join(vars_conjuction)}/SV_{score_ver}"
        os.makedirs(sdir_base, exist_ok=True)
        print("Saving to:", sdir_base)

        # Focus on the higher firing rate sites
        fig = self.sites_update_fr_thresh(FR_THRESH, FR_PERCENTILE, True, True)
        fig.savefig(f"{sdir_base}/sites_pruned_by_fr_hist.pdf")
        #plt.close(fig)

        ###### GLOBALS
        if PRE_DUR_CALC is None:
            assert list_pre_dur is not None
        else:
            assert list_pre_dur is None

        self.globals_initialize()
        self.globals_update(globals_nmin, 
            globals_lenient_allow_data_if_has_n_levels, 
            PRE_DUR_CALC, POST_DUR_CALC, list_events,
            list_pre_dur, list_post_dur)

        ##############################################
        # Get conjunctions and print
        if vars_conjuction is None:
            _list_vars = [var]
        else:
            _list_vars = [var] + vars_conjuction
        self.modulationgood_plot_list_conjuctions(_list_vars, SAVEDIR=sdir_base) 
        if do_only_print_conjunctions:
            print("DONE! just printed conjunctions, as you requested... (exiting..)")
            return

        # IF there is no variation in var, then exit
        if len(self.DfScalar[var].unique().tolist())==1:
            print("SKIPPING, this var only has one level:")
            print(self.DfScalar[var].unique().tolist())
            raise NotEnoughDataException

        # self.ParamsGlobals["PRE_DUR_CALC"] = PRE_DUR_CALC
        # self.ParamsGlobals["POST_DUR_CALC"] = POST_DUR_CALC
        sn, _ = self._session_extract_sn_and_trial()


        ######### SAVE PARAMS
        path = f"{sdir_base}/Params.yaml"
        writeDictToYaml(self.Params, path)

        path = f"{sdir_base}/ParamsGlobals.yaml"
        writeDictToYaml(self.ParamsGlobals, path)

        if params_to_save is not None:            
            path = f"{sdir_base}/params_to_save.yaml"
            writeDictToYaml(params_to_save, path)

        if False:
            # Skip this, slows things down, including the plots.
            # OBSOLETE!!
            print("** Computing fr quick...")
            df_fr, df_fr_levels = self.modulationgood_compute_fr_quick(var)
            # df_fr = df_fr[df_fr["event_aligned"] == event]
            # df_fr_levels = df_fr_levels[df_fr_levels["event_aligned"] == event]
            assert len(df_fr)>0
            assert len(df_fr_levels)>0
        else:
            df_fr = None

        ################# LOAD DF_VAR
        df_var, list_eventwindow_event, RELOAD_DFVAR_SUCCESSFUL, RECOMPUTED = self.modulationgood_load_or_compute(
            "df_var", sdir_base, var, vars_conjuction, score_ver, get_z_score)

        df_fr_levels, _, RELOAD_DFVAR_SUCCESSFUL, RECOMPUTED = self.modulationgood_load_or_compute(
            "df_fr_levels", sdir_base, var, vars_conjuction, None, None)

        ################# save again
        path = f"{sdir_base}/ParamsGlobals.yaml"
        writeDictToYaml(self.ParamsGlobals, path)

        list_events_window = sorted(df_var["event"].unique().tolist())


        # Save df_var
        if False:
            # Not using this anymore, doing inside modulationgood_load_or_compute
            path = f"{sdir_base}/df_fr.pkl"
            print("SAving: ", path)
            with open(path, "wb") as f:
                pickle.dump(df_fr, f)

            # Save df_var
            path = f"{sdir_base}/df_fr_levels.pkl"
            print("SAving: ", path)
            with open(path, "wb") as f:
                pickle.dump(df_fr_levels, f)

        #################### QUICK, plots of fr vs. levels
        if df_fr_levels is not None and len(df_fr_levels)>0:
            sdir = f"{sdir_base}/fr_levels"
            os.makedirs(sdir, exist_ok=True)
            print("Plotting... ", sdir)
            self.modulationgood_plot_summarystats_fr(None, df_fr_levels, savedir=sdir)
            plt.close("all")

        ################### MAIN PLOTS OF MODULATION
        # Only do plots if did recomputed
        if RECOMPUTED and len(df_var)>0:
            # How many ways anova?
            if sum(df_var["val_kind"]=="val_others")>0 and sum(df_var["val_kind"]=="val_interaction")>0:
                # two way anova
                N_WAYS = 2
            else:
                N_WAYS = 1

            paramstmp = {
                "anova_n_ways":N_WAYS,
                "var":var,
                "vars_conjuction":vars_conjuction,
                "list_events":list_events,
                "list_events_window":list_events_window,
                "score_ver":score_ver,
                "SAVEDIR":SAVEDIR,
                "PRE_DUR_CALC":PRE_DUR_CALC,
                "POST_DUR_CALC":POST_DUR_CALC,
                "globals_nmin":globals_nmin,
                "globals_lenient_allow_data_if_has_n_levels":globals_lenient_allow_data_if_has_n_levels,
                "FR_THRESH":FR_THRESH,
                "get_z_score":get_z_score,
                "ANALY_VER":ANALY_VER,
                "supervision_keep_only_not_training":supervision_keep_only_not_training,
                "trialcodes_keep":trialcodes_keep,
                "df_scalar_pruned_length":df_scalar_pruned_length,
                "df_scalar_orig_length":df_scalar_orig_length
                }
            path = f"{sdir_base}/params_modulationgood_compute.yaml"
            writeDictToYaml(paramstmp, path)
            self.ParamsDfvar = paramstmp

            ############## SUMMARY PLOTS OF ANOVA
            self.modulationgood_plot_WRAPPER(df_var, None, None,
                list_eventwindow_event, var, vars_conjuction, 
                sdir_base, N_WAYS, PLOT_EACH_CHAN, PLOT_EACH_EVENT)

            ##### Plot example strokes extracted. Group them by levels of var
            # if self.DS is not None:
            print("** Plotting example strokes")
            sdir = f"{sdir_base}/drawings"
            os.makedirs(sdir, exist_ok=True)
            self.modulationgood_plot_drawings_variables(var, vars_conjuction, sdir)

        else:
            print("********* SKIPPING PLOTTING OF DF_VAR (since did not recompute df_var")
            
        ##### Plot rasters
        if PLOT_RASTERS:
            for event_window, event in list_eventwindow_event:
                # (Only do once for each event)
                sdir_rasters = f"{SAVEDIR}/{ANALY_VER}/var_by_varsother/VAR_{var}-OV_{'_'.join(vars_conjuction)}/rasters/{event}"
                os.makedirs(sdir_rasters, exist_ok=True)

                print("** Plotting raster + sm fr:", sdir_rasters)
                ##### Plot raster + sm fr
                # Plot rasters for each site
                old_backend = mpl.get_backend()
                print("default backend is " + old_backend)
                mpl.use('agg') # non-GUI backend, so that the loop below doesn't run into a memory leak error (see GitHub matplotlib: #20300)
                for site in self.Sites:
                    path = f"{sdir_rasters}/{sn.sitegetter_summarytext(site)}.png"
                    if not os.path.exists(path):
                        fig, axes = self.plotgood_rasters_smfr_each_level_combined(site, var, vars_conjuction, 
                            event=event)
                        fig.savefig(path)
                        plt.close("all")
                mpl.use(old_backend) # switch back just to be safe..

        else:
            print("!!SKIPPING PLOTS!! not enough data")
            print(len(df_var))


    def modulationgood_load_or_compute(self, which_data, sdir_base, var, 
                    vars_conjuction, score_ver, get_z_score,
                    DEBUG_CONJUNCTIONS=False,
                    RESAVE_AND_MOVE_OLD_PLOTS=True):
        """ Helper to try to load df_var, and then compute any new events taht are
        not present, and then concatenate and then save again
        PARAMS:
        - which_data, string, 
        - vars_conjuction, list
        - RESAVE_AND_MOVE_OLD_PLOTS, bool, if True, then resaved the eextracted d, and moves the old data
        into a subdir.Otherwise just returns the data
        RETURNS:
        - DFTHIS, either df_var, or df_fr_levels, depending on which_data
        - list_eventwindow_event, list of tuple, each is (event with exct time window, num_event)
        """
 
        from pickle import UnpicklingError
        from pythonlib.tools.expttools import load_yaml_config

        if which_data=="df_var":
            FNAME = "df_var"
            THINGS_TO_EXTRACT = tuple(["anova"])
            IDX = 0
        elif which_data=="df_fr_levels":
            FNAME = "df_fr_levels"
            THINGS_TO_EXTRACT = tuple(["fr"])
            IDX = 2
        else:
            print(which_data)
            assert False

        #### TRY LOADING Check if df is already saved
        EVENTS_ALREADY_DONE = []
        RELOAD_DFVAR_SUCCESSFUL = False
        path = f"{sdir_base}/{FNAME}.pkl"
        print(f"Searching for already-done {which_data} at this path:", path)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    DFTHIS_SAVED = pickle.load(f)
                print(f"RELOADED {which_data}!!!")

                list_eventwindow_event_saved = sorted(set([tuple(x) for x in DFTHIS_SAVED.loc[:, ["event", "_event"]].values.tolist()]))

                # Only recompute events which you have not already extradcted.
                RELOAD_DFVAR_SUCCESSFUL = True
                EVENTS_ALREADY_DONE = [x[0] for x in list_eventwindow_event_saved] # er.g, [02_rulecue2_40_to_600]
                print("Events already done: (will skip these when recomputing)...")
                print(EVENTS_ALREADY_DONE)
            except UnpicklingError as err:
                print(f"Corrupted {FNAME}.pkl --> Recomputing!")
        else:
            print("... path does not exist")

        print(f"COMPUTING {which_data}!!!")
        try:
            print(EVENTS_ALREADY_DONE)
            print(THINGS_TO_EXTRACT)
            res = self.modulationgood_compute_wrapper(var, 
                vars_conjuction, score_ver=score_ver, get_z_score=get_z_score,
                DEBUG_CONJUNCTIONS=DEBUG_CONJUNCTIONS,
                events_windowed_skip = EVENTS_ALREADY_DONE, SAVEDIR=sdir_base,
                THINGS_TO_EXTRACT=THINGS_TO_EXTRACT)
            DFTHIS = res[IDX]
            list_eventwindow_event = res[3]

            if len(DFTHIS)>0:
                RECOMPUTED=True
            else:
                RECOMPUTED = False

            if RELOAD_DFVAR_SUCCESSFUL:
                # Then you want to merge
                print("... Merging pre-saved and new df!")
                print("-- Len of old df, and events that exist in it:")
                print(len(DFTHIS_SAVED))
                print(DFTHIS_SAVED["event"].unique())
                print("-- Len of new df, and events that exist in it:")
                print(len(DFTHIS))
                print(DFTHIS["event"].unique())

                DFTHIS = pd.concat([DFTHIS, DFTHIS_SAVED], axis=0).reset_index(drop=True)
                list_eventwindow_event.extend(list_eventwindow_event_saved)
            # else:
            #     print(".. computing for the first time is DONE!")
            #     DFTHIS = DFTHIS_SAVED
            #     list_eventwindow_event = list_eventwindow_event_saved

        except NotEnoughDataException as err:
            print("--NotEnoughDataException [modulationgood_compute_wrapper] during: ", var, vars_conjuction, score_ver)
            RECOMPUTED = False
            if RELOAD_DFVAR_SUCCESSFUL:
                print("This is ok, since you already have pre-saved df loaded... continuing with loaded...")
                DFTHIS = DFTHIS_SAVED
                list_eventwindow_event = list_eventwindow_event_saved
            else:
                print("ACutlaly a failure, you dont have presaved, and could not extract new... raising error")
                # print("Now rerunning modulationgood_compute_wrapper, with DEBUG_CONJUNCTIONS=True...")
                # df_var, list_eventwindow_event = self.modulationgood_compute_wrapper(var, 
                #     vars_conjuction, score_ver=score_ver, get_z_score=get_z_score,
                #     DEBUG_CONJUNCTIONS=True,
                #     events_windowed_skip = EVENTS_ALREADY_DONE, SAVEDIR=sdir_base) 
                raise err

        print("-- Len of FINAL df_var, and events that exist in it:")
        print(len(DFTHIS))
        print(DFTHIS["event"].unique())

        ################ CLEAN AND SAVE
        DFTHIS = DFTHIS.reset_index(drop=True)
        # list_events_window = sorted(DFTHIS["event"].unique().tolist()) 

        if RESAVE_AND_MOVE_OLD_PLOTS and RECOMPUTED==True:

            ##### Move old plots
            if RELOAD_DFVAR_SUCCESSFUL:
                ######### MOVE OLD DF_VAR AND OLD PLOTS
                from pythonlib.tools.expttools import makeTimeStamp
                import glob
                import shutil
                from pythonlib.tools.expttools import fileparts
                ts = makeTimeStamp()
                items = glob.glob(f"{sdir_base}/*")
                dir_move = f"{sdir_base}/OLD_PLOTS-moved_on_{ts}"
                os.makedirs(dir_move, exist_ok=True)
                for it in items:
                    fname = fileparts(it)[-2] + fileparts(it)[-1] # e.g., df_var.pkl
                    path_new = f"{dir_move}/{fname}"
                    print(f"Moving {it} to {path_new}")
                    shutil.move(it, path_new)
                print("All files moved... ready to remake plots!!")

            ############ Save df_var
            path = f"{sdir_base}/{FNAME}.pkl"
            print("SAving: ", path)
            with open(path, "wb") as f:
                pickle.dump(DFTHIS, f)

            # Save df_var
            path = f"{sdir_base}/list_eventwindow_event.pkl"
            print("SAving: ", path)
            with open(path, "wb") as f:
                pickle.dump(list_eventwindow_event, f)

        ### ATTACH SOME PARAMS TO SP
        self.ParamsDictDfvar = {}
        if which_data == "df_var":    
            params_path = f"{sdir_base}/params_modulationgood_compute.yaml"
            if os.path.exists(params_path):
                self.ParamsDictDfvar["params_modulationgood_compute"] = load_yaml_config(params_path)
                self.ParamsDfvar = load_yaml_config(params_path)

            params_path = f"{sdir_base}/ParamsGlobals.yaml"
            if os.path.exists(params_path):
                self.ParamsDictDfvar["ParamsGlobals"] = load_yaml_config(params_path)

            params_path = f"{sdir_base}/params_to_save.yaml"
            if os.path.exists(params_path):
                self.ParamsDictDfvar["params_to_save"] = load_yaml_config(params_path)

            params_path = f"{sdir_base}/Params.yaml"
            if os.path.exists(params_path):
                self.ParamsDictDfvar["Params"] = load_yaml_config(params_path)

        return DFTHIS, list_eventwindow_event, RELOAD_DFVAR_SUCCESSFUL, RECOMPUTED


    def modulationgood_compute_wrapper(self, var, vars_conjuction=None, list_site=None, 
            score_ver="r2smfr_minshuff", get_z_score=False,
            DEBUG_CONJUNCTIONS=False,
            events_windowed_skip = None, SAVEDIR=None,
            THINGS_TO_EXTRACT=("fr", "anova")):
        """ Good, flexible helper to compute modulation of all kinds and all ways of slicing 
        the dataset. 
        PARAMS;
        - n_min, min n trials required for each level of var. if faisl, then skips this datset 
        entirely (i.e., the level of vars_conjuction, or this site)
        - lenient_allow_data_if_has_n_levels, eitehr None (ignore) or int, how many
        levels of var you need to get >n_min datapts, in order to keep this level of vars_conj./
        See within for detials.
        - events_windowed_skip, either None(ignores) or list of str, each an event to skip if 
        it comes up during analy, e.g, [02_rulecue2_40_to_600]
        RETURNS:
        - DF_VAR, 
        - DF_FR, [IGNORE THIS]
        - DF_FR_LEVELS
        - list_eventwindow_event, list of tuples, each a (event_window, event), whgere ew is conjucntion
        of event and (predur, postdur) and event is the original event name. such as :
            [('00_fix_touch_-500_to_0', '00_fix_touch'),
             ('01_samp_50_to_600', '01_samp'),
             ('03_first_raise_-100_to_500', '03_first_raise')]
        """
        from pythonlib.tools.pandastools import append_col_with_grp_index
        import numpy as np
        from pythonlib.tools.pandastools import grouping_print_n_samples

        # Want to use entier data for this site? or do separately for each level of a given
        # conjunction variable.
        if True:
            # nio need to do. its done auto in dataextract_as_df_conjunction_vars
            if vars_conjuction is None:
                # then place a dummy variable so that entire thing is one level
                vars_conjuction = ["dummy_var"]
                # delete it
                # self.DfScalar = self.DfScalar.drop("dummy_var", axis=1)
                # assert "dummy_var" not in self.DfScalar.columns
                self.DfScalar["dummy_var"] = "dummy"
                # vars_conjuction = ['gridloc', 'chunk_within_rank'] # list of str, vars to take conjunction over

        ##### First, skip everything if there is not enough data
        list_grp = ["chan", "event", var]
        if vars_conjuction:
            list_grp = list_grp + vars_conjuction

        print("Running grouping_print_n_samples...")
        if False:
            groupdict = grouping_print_n_samples(self.DfScalar, list_grp)
            n_max = max([n for n in groupdict.values()])
        else:
            try:
                from pythonlib.tools.pandastools import grouping_count_n_samples_quick
                n_min, n_max = grouping_count_n_samples_quick(self.DfScalar, list_grp)
            except KeyError as err:
                print("SKipping, you dont have a variable in self.DfScalar. Need to rerun extraction..")
                print("These are the existing columns")
                print(self.DfScalar.columns)
                print("failed searchign for these columns:")
                print(list_grp)
                raise NotEnoughDataException
            except Exception as err:
                print("These are the existing columns")
                print(self.DfScalar.columns)
                print("failed searchign for these columns:")
                print(list_grp)
                raise err

        if list_site is None:
            list_site = self.Sites

        # Extract globals for analys
        list_events = self.ParamsGlobals["list_events"]
        list_pre_dur = self.ParamsGlobals["list_pre_dur"]
        list_post_dur = self.ParamsGlobals["list_post_dur"]
        print("DOing these! ...")
        print("list_events", list_events)
        print("list_pre_dur", list_pre_dur)
        print("list_post_dur", list_post_dur)
        
        if events_windowed_skip is not None:
            print("WILL SKIP THESE EVENTS...")
            print(events_windowed_skip)
        else:
            events_windowed_skip = []

        sn, _ = self._session_extract_sn_and_trial()

        if n_max<self.ParamsGlobals["n_min_trials_per_level"]:
            print("SKIPPING PREEMPTIVELY, not enough data. (chan, event, vars, othervars) max n = ", n_max)
            print("... which is lower than ", self.ParamsGlobals['n_min_trials_per_level'])
            raise NotEnoughDataException
        else:
            print("GOOD!, enough data, max n per grouping conjunction (nmin, nmax) ", n_min, n_max)

        if score_ver in ["r2smfr_running_maxtime_twoway", "r2_maxtime_2way_mshuff"]:
            TWO_WAY=True
            assert vars_conjuction is not None
            # check that the othervars have >1 level
            n_lev_othervar = len(self.DfScalar.groupby(vars_conjuction))
            if n_lev_othervar<2:
                print("!!SKIPPING PREEMPTIVELY, since is two-way but only have <2 levels for othervars")
                raise NotEnoughDataException
        else:
            TWO_WAY=False

        # Collect data for each site
        OUT = []
        OUT_FR = []
        OUT_FR_LEVELS = []
        list_eventwindow_event = []
        for event, pre_dur, post_dur in zip(list_events, list_pre_dur, list_post_dur):
            
            ################################################
            ################# UDPATE GLOBAL PARAMS!!
            print(" ")
            print(f"Updated ParamsGlobals for event {event} to:")
            self.globals_update(PRE_DUR_CALC=pre_dur, POST_DUR_CALC=post_dur)
            event_window_combo_name = f"{event}_{pre_dur*1000:.0f}_to_{post_dur*1000:.0f}"
            print("DOING THIS EVENT: ", event_window_combo_name)

            if event_window_combo_name in events_windowed_skip:
                print("!!!! SKIPPING this event, since it is in events_windowed_skip that you entered:")
                print(event_window_combo_name)
                continue

            _saved_counts = False
            sites_exist = self.DfScalar["chan"].tolist()
            for site in list_site:

                if site not in sites_exist:
                    continue

                if site%20==0:
                    print("site :", site)
                region = sn.sitegetterKS_map_site_to_region(site)
                
                # Clean up dataset
                # print(var)
                # print(vars_conjuction)
                # print(site)
                # print(event)
                dfthis, levs_df, levels_var = self.dataextract_as_df_conjunction_vars(var, 
                    vars_conjuction, site, event=event, DEBUG_CONJUNCTIONS=DEBUG_CONJUNCTIONS)
                # print(len(dfthis))                
                # assert False
                if len(dfthis)==0:
                    print("SKIPPING ", event, site)
                    continue

                if "fr" in THINGS_TO_EXTRACT:
                    ############### MEAN FR for each (chan, event, var, othervar)
                    # 1) Reextract FR (since this is a new globals param)
                    dfthis = self._modulationgood_extract_fr(dfthis=dfthis)

                    # 2) Compute
                    df_fr, df_fr_levels = self.modulationgood_compute_fr_quick(var,
                        "vars_others", dfthis=dfthis) 

                    # append this window
                    df_fr["_event"] = df_fr["event"]
                    df_fr["event"] = event_window_combo_name
                    df_fr["event_aligned"] = event_window_combo_name
                    
                    df_fr_levels["_event"] = df_fr_levels["event"]
                    df_fr_levels["event"] = event_window_combo_name
                    df_fr_levels["event_aligned"] = event_window_combo_name

                    # 3) Collect
                    OUT_FR.append(df_fr)
                    OUT_FR_LEVELS.append(df_fr_levels)

                if "anova" in THINGS_TO_EXTRACT:
                    ################## ANOVA
                    # v2) two-way anova
                    if TWO_WAY:
                            
                        if len(levs_df)<2:
                            # You need at least 2 levels for othervers to run two-way anova
                            print("!!SKIPPING, since is two-way but only have <2 levels for othervars")
                            continue

                        MS = self._dataextract_as_metrics_scalar(dfthis, var)    

                        # try:
                        assert get_z_score==False, "not yet coded, this gets messy, one z for var, var_others, interaction"
                        eventscores, dfcounts = MS.modulationgood_wrapper_twoway(var, version=score_ver, 
                            vars_others = "vars_others", auto_balance_conjunctions_exist=True)
                        assert len(eventscores.keys())==1, "should only be one event"
                        # except ValueWarning as err:
                        #     print("HERE")
                        #     print(len(MS.Data))
                        #     print("HERE")
                        #     print(site, lev)
                        #     print("HERE")
                        #     print(MS.Data[var].value_counts())
                        #     raise err
                        #     assert False

                        score, score_others, score_interaction = eventscores[event]

                        assert isinstance(score, float), "no np arrays allowed. will fail seaborn"
                        assert isinstance(score_others, float), "no np arrays allowed. will fail seaborn"
                        assert isinstance(score_interaction, float), "no np arrays allowed. will fail seaborn"

                        # Save csv of n data, to know how much pruning was done...
                        # (to balance the data)
                        # Only plot the first one, since they should be same across sites
                        if _saved_counts==False:
                            dirthis = f"{SAVEDIR}/twoway_pruned_balance_samplesizes"
                            os.makedirs(dirthis, exist_ok=True)
                            path = f"{dirthis}/{event_window_combo_name}-{site}.csv"
                            dfcounts.to_csv(path)
                            _saved_counts = True

                        # save
                        for val_kind, val in zip(["val", "val_others", "val_interaction"], [score, score_others, score_interaction]):
                            OUT.append({
                                "chan":site,
                                "var":var,
                                "var_others":tuple(vars_conjuction),
                                "_event":event,
                                "event":event_window_combo_name,                            
                                "val_kind":val_kind,
                                "val_method":score_ver,
                                "val":val,
                                "bregion":region,
                                "n_datapts":len(MS.Data)
                            })


                    else:
                        # TWO_WAY = False
                        # if len(dfthis)==0:
                        #     # then no level of vars_conjuction had enough data across all levels of var.
                        #     continue

                        # for each level of vars_conj, compute modulation
                        # levels_others = dfthis["vars_others"].unique().tolist()
                        for lev, dfthisthis in levs_df.items():
                        # for lev in levels_others:
                            
                        #     assert len(lev)==len(vars_conjuction)
                            
                        #     # get data
                        #     dfthisthis = dfthis[dfthis["vars_others"]==lev] # specific df (lev_other)
                        #     assert len(dfthisthis)>0

                            # v1) keeping fr for each level of each var
                            # if score_ver in ["fr_chan_level"]:
                            #     assert False, "in progress... just need to modify MS to take in var and levels."
                            #     # Then is a single score for each level of var
                            #     MS = self._dataextract_as_metrics_scalar(dfthisthis, var, event=event)
                            #     tmp = MS.calc_fr_across_levels()
                            #     print(tmp)
                            #     assert False

                            #     # frates = _calc_fr_across_levels(dfthisthis, var, levels_var)["all_data"]
                                
                            #     for lv, fr in zip(levels_var, frates):
                            #         OUT.append({
                            #             "chan":site,
                            #             "var":var,
                            #             "lev_in_var":lv,                            
                            #             "var_others":tuple(vars_conjuction),
                            #             "lev_in_var_others":lev,
                            #             "event":ev,
                            #             "val_kind":"modulation_subgroups",
                            #             "val_method":score_ver,
                            #             "val":score,
                            #             "bregion":region
                            #         })
                            #         # also save columns for each var in vars_others
                            #         for l, v in zip(lev, vars_conjuction):
                            #             OUT[-1][v] = l

                            # v3) one-way anova (but doing separately for each level of othervars)
                            # Then is a single score per (chan, event, var)
                            # compute modulation
                            # - one value for each var.
                            MS = self._dataextract_as_metrics_scalar(dfthisthis, var) 

                            # try:
                            eventscores = MS.modulationgood_wrapper_(var, version=score_ver, 
                                return_as_score_zscore_tuple=get_z_score)
                            # except ValueWarning as err:
                            #     print("HERE")
                            #     print(len(MS.Data))
                            #     print("HERE")
                            #     print(site, lev)
                            #     print("HERE")
                            #     print(MS.Data[var].value_counts())
                            #     raise err
                            print(eventscores.keys())
                            assert len(eventscores.keys())==1, "should only be one event"
                            score = eventscores[event]

                            if get_z_score:
                                score, zscore = score
                            else:
                                zscore = np.nan

                            assert isinstance(score, float), "no np arrays allowed. will fail seaborn"
                            assert isinstance(zscore, float), "no np arrays allowed. will fail seaborn"

                            # save
                            OUT.append({
                                "chan":site,
                                "var":var,
                                "var_others":tuple(vars_conjuction),
                                "lev_in_var_others":lev,
                                "_event":event,
                                "event":event_window_combo_name,                            
                                "val_kind":"modulation_subgroups",
                                "val_method":score_ver,
                                "val":score,
                                "val_zscore":zscore,
                                "bregion":region,
                                "n_datapts":len(MS.Data)
                            })

                            # also save columns for each var in vars_others
                            for l, v in zip(lev, vars_conjuction):
                                OUT[-1][v] = l

            list_eventwindow_event.append((event_window_combo_name, event))
            

        ##################### CONCATE INTO OUTPUT DATA
        if "fr" in THINGS_TO_EXTRACT:
            if len(OUT_FR_LEVELS)==0:
                print("SKIPPING, extracted DF_FR_LEVELS is empty. Probably you have not enough data for this conjunctions, try setting DEBUG_CONJUNCTIONS=True and reading the low-level data it prints.")
                raise NotEnoughDataException            
                
            DF_FR = pd.concat(OUT_FR).reset_index(drop=True)
            DF_FR_LEVELS = pd.concat(OUT_FR_LEVELS).reset_index(drop=True)
        else:
            DF_FR = None
            DF_FR_LEVELS = None

        if "anova" in THINGS_TO_EXTRACT:
            DF_VAR = pd.DataFrame(OUT)      
            if len(DF_VAR)==0:
                print("SKIPPING, extracted df_var is empty. Probably you have not enough data for this conjunctions, try setting DEBUG_CONJUNCTIONS=True and reading the low-level data it prints.")
                raise NotEnoughDataException
        else:
            DF_VAR = None

        ################### CLEANUP
        if False:
            # Keep it, if no conjucntion vars inputed, dummy_var is required.
            if "dummy_var" in self.DfScalar:
                del self.DfScalar["dummy_var"]

        # Melt, if this is anova with multiple levels. such that "val_kind" holds var, other, and interaction.
        if False: # Becuase this was failing. cant get uniqque values for numericals
            if TWO_WAY:
                from pythonlib.tools.pandastools import unpivot
                print(DF_VAR.columns)
                DF_VAR = unpivot(DF_VAR, id_vars=["chan", "var", "var_others", "_event", "event", "val_kind", "val_method", "bregion", "n_datapts"], 
                        value_vars=["val", "val_others", "val_interaction"], var_name="val_kind", value_name="val")

                print(DF_VAR.columns)

        if False:
            list_eventwindow_event = sort_mixed_type(set([tuple(x) for x in DF_VAR.loc[:, ["event", "_event"]].values.tolist()]))
            # [('00_fix_touch_-500_to_0', '00_fix_touch'),
            #  ('01_samp_50_to_600', '01_samp'),
            #  ('03_first_raise_-100_to_500', '03_first_raise')]#

        return DF_VAR, DF_FR, DF_FR_LEVELS, list_eventwindow_event

    def modulationgood_aggregate_df(self, df_var, aggmethod="weighted_avg"):
        """ Wrapper for methods for aggregating data, using so there's single datapt
        per chan. 
        """
        from pythonlib.tools.pandastools import aggregThenReassignToNewColumn, aggregGeneral
        # AGGMETHOD = "max"

        GROUPING = ["chan", "event", "var", "var_others", "val_kind", "val_method"]
        if aggmethod=="weighted_avg":
            # Take weighted average, where weights are based on n ttrials
            # - avg of scores
            def F(x):
                try:
                    vals = x["val"]
                    ndats = x["n_datapts"]
                    weights = (ndats/np.mean(ndats))**0.5
                    tmp = np.average(vals, weights=weights)
                except Exception as err:
                    print(x["val"].tolist())
                    print(x["n_datapts"].tolist())
                    raise err
                return tmp
            df_var = aggregThenReassignToNewColumn(df_var, F, GROUPING, "val_weighted_avg")

            if "val_zscore" in df_var.columns:
                def F(x):
                    vals = x["val_zscore"]
                    ndats = x["n_datapts"]
                    weights = (ndats/np.mean(ndats))**0.5
                    try:
                        tmp = np.average(vals, weights=weights)
                    except Exception as err:
                        print(x["val_zscore"].tolist())
                        print(x["n_datapts"].tolist())
                        raise err
                    return tmp
                df_var = aggregThenReassignToNewColumn(df_var, F, GROUPING, "val_z_weighted_avg")

                # - Do aggregation
                dfout_agg = aggregGeneral(df_var, group=GROUPING, values=["val_weighted_avg", "val_z_weighted_avg"])
                dfout_agg["val"] = dfout_agg["val_weighted_avg"]
                dfout_agg["val_zscore"] = dfout_agg["val_z_weighted_avg"]
                dfout_agg = dfout_agg.drop("val_weighted_avg", axis=1)
                dfout_agg = dfout_agg.drop("val_z_weighted_avg", axis=1)
            else:
                # This is probably 2-way anova, or other analyses, which did not have z-score
                dfout_agg = aggregGeneral(df_var, group=GROUPING, values=["val_weighted_avg"])
                dfout_agg["val"] = dfout_agg["val_weighted_avg"]
                dfout_agg = dfout_agg.drop("val_weighted_avg", axis=1)
        elif aggmethod=="max":
            assert False, "in p[rogess by basiclaly done, below."
            # TAKING, using zsocre to get argmax. 
            # # take value for case with max z-score
            # def F(x):
            #     vals = x["val"].tolist()
            #     vals_z = x["val_zscore"].tolist()
            #     tmp = vals[np.argmax(vals_z)]
            #     return tmp
            # df_var = aggregThenReassignToNewColumn(df_var, F, ["chan"], "val_max", return_grouped_df=False)

            # # take value for case with max z-score (also the z)
            # def F(x):
            #     vals = x["val"].tolist()
            #     vals_z = x["val_zscore"].tolist()
            #     return np.max(vals_z)
            # df_var = aggregThenReassignToNewColumn(df_var, F, ["chan"], "val_zscore_max", return_grouped_df=False)


            # dfout_agg = aggregGeneral(df_var, group=["chan", "event"], values=["val_max", "val_zscore_max"])

            # dfout_agg["val"] = dfout_agg["val_max"]
            # dfout_agg["val_zscore"] = dfout_agg["val_zscore_max"]
        elif aggmethod=="n_sig_cases_othervars":
            # Distribution of number of cases significant. i.e for each channel
            # get how many specific conjucntions of "other vars" have sign.
            # modulation by var, where this is based on thresholding the z-score.
            thresh_zscore = 3
            res = []
            for site in self.Sites:
                for ev in df_var["event"].unique().tolist():
                    dfthis = df_var[(df_var["chan"]==site) & (df_var["event"]==ev)]
                    
                    nsig = sum(dfthis["val_zscore"]>=thresh_zscore)
                    ntot = len(dfthis)
                    
                    res.append({
                        "event":ev,
                        "chan":site,
                        "nsig":nsig,
                        "ntot":ntot
                    })
                    
            dfout_agg = pd.DataFrame(res)


        # APpend bregion
        dfout_agg = self.datamod_append_bregion(dfout_agg)

        return dfout_agg



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
            regions_in_order = self.SN.sitegetter_get_brainregion_list_BASE(bregion_combine)
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
            
    def modulationgood_plot_each_chan(self, df_var, var, vars_conjuction,
        savedir="/tmp"):
        """ Plot for each chan, overview of modulation across all variables, 
        and alll events
        """
        import seaborn as sns

        list_var = [var]
        list_events_uniqnames = self.Params["list_events_uniqnames"]
        sites_keep = self.Sites

        # # good [OLD. replaced by below]]
        # fig, ax = plt.subplots()
        # for site in SP.Sites:
        # #     sns.pointplot(data=dfout, x="event", y="val", hue="lev_in_var_others", row="var", col="var_others")
        #     dfthis = dfout[dfout["chan"]==site]
        #     fig = sns.catplot(data=dfout, x="event", y="val", hue="lev_in_var_others", row="var", kind="point", ci=68)
            
        #     fig.savefig(f"{sdir}/{SP.SN.sitegetter_summarytext(site)}-modulation.pdf")
        #     assert False

        for site in sites_keep:

            # get data for this site.
            dfthis, _, levels_var = self.dataextract_as_df_conjunction_vars(var, 
                vars_conjuction, site)
            
            # ymax = dfthis["fr_scalar"].max()

            print("PLotting for (chan): ", site)

        #     ##################### Plot separately each var (showing its modulation)
        #     if len(list_var)>2:
        #         # otherwise doesnt make sense, this is all captured int he overview plot.
        #         for xvar in list_var:
        #             _, var_hue, var_col = _find_varhue_varcol(xvar, list_var)

        #             fig = sns.catplot(data=dfthis, x=xvar, y="fr_scalar", hue=var_hue, 
        #                 row="event_aligned", col=var_col, kind="point", height=3)
        #             rotateLabel(fig)

        #             # fr, scale from 0
        #             for ax in fig.axes.flatten():
        #                 ax.set_ylim([0, ymax])

        #             # Save
        #             fig.savefig(f"{savedir}/{bregion}-{chan}-x_{xvar}.pdf")

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
                    g = sns.pointplot(ax=ax, data=dfthisthis, x=var, y="fr_scalar", hue="vars_others")
                    if i>0:
                        # only keep legend for first row
                        g.legend().remove()        

                    # fr, scale from 0
                    # ax.set_ylim([0, ymax])
                    ax.set_ylim(0)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

                    if j==1:
                        ax.set_ylabel(var)
                    if i==0:
                        ax.legend(framealpha=0.5)


            ### 2) Average fr across all (other var levels)
            ax = axes[len(list_events_uniqnames)][j]
            sns.pointplot(ax=ax, data=dfthis, x=var, y = "fr_scalar", hue="event_aligned")

            # fr, scale from 0
            # ax.set_ylim([0, ymax])
            ax.set_ylim(0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
            ax.legend(framealpha=0.5)

            ### 3) Modulation vs. event
            ax = axes[len(list_events_uniqnames)][ncols-1]
            ax2 = axes[len(list_events_uniqnames)-1][ncols-1]
            
            dftmp = df_var[df_var["chan"]==site]
            sns.pointplot(ax=ax2, data=dftmp, x="event", y="val", hue="lev_in_var_others", row="var", col="var_others")
        #     fig = sns.catplot(data=dfout, x="event", y="val", hue="lev_in_var_others", row="var", kind="point", ci=68)
            ax2.set_title("modulation")
            ax2.set_ylim(0)
        #     pcols = makeColors(len(list_var))
        #     for var, pcol in zip(list_var, pcols):

        #         vals = RES["modulation_across_events"][var]
        #         ax.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)
        #         ax2.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)

        #         vals = RES["modulation_across_events_subgroups"][var]
        #         ax.plot(list_events_uniqnames, vals, '--o', color=pcol, label=f"{var}_othervars_mean")
        #         ax2.plot(list_events_uniqnames, vals, '-o', color=pcol, label=var)

            ### 3) mean fr across events (simple)
            ax = axes[1][ncols-1]
            sns.pointplot(ax=ax, data=dfthis, x="event_aligned", y = "fr_scalar")
            # fr, scale from 0
            # ax.set_ylim([0, ymax])
            ax.set_ylim(0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
            ax.set_title("firing rate")


            ### 4) Save4
            sn, _ = self._session_extract_sn_and_trial()
            st = sn.sitegetter_summarytext(site)
            fig.savefig(f"{savedir}/{st}-overview.pdf")
            plt.close("all")

    def modulationgood_plot_drawings_variables(self, var, vars_conjuction, sdir,
            nplot = 40):
        """
        Plot example drawings across lewvels for a given variable, seaprate
        figure for each level of the conjucntion variaable.
        """

        if self.Params["which_level"] in ["trial", "flex"]:
            plotfunc = lambda var, vars_conjuction, sdir, nplot: self._modulationgood_plot_drawings_variables_bytrial(var, vars_conjuction, sdir, nplot)
        elif self.Params["which_level"]=="stroke":
            plotfunc = lambda var, vars_conjuction, sdir, nplot: self._modulationgood_plot_drawings_variables(var, vars_conjuction, sdir, nplot)
        else: 
            #plotfunc = lambda var, vars_conjuction, sdir, nplot: self._modulationgood_plot_drawings_variables_bytrial(var, vars_conjuction, sdir, nplot)
            assert False

        # First, is this trial or stroke level.
        if self.SNmult is None:
            # Then this is a single session, plot it.
            if self.Params["which_level"] in ["trial", "flex"]:
                return self._modulationgood_plot_drawings_variables_bytrial(var, vars_conjuction, sdir, nplot)
            else:
                return self._modulationgood_plot_drawings_variables(var, vars_conjuction, sdir, nplot)
        else:
            # Load each session, and plot it.
            for idx_session in self.DfScalar["session_idx"].unique().tolist():
                
                sn = self.SNmult.SessionsList[idx_session]
                sp = load_snippet_single(sn, which_level=self.Params["which_level"])

                # replace sp with pruned trials according to self. Makes suer to only plot included trials.
                df = self.DfScalar[self.DfScalar["session_idx"] == idx_session]
                sp.DfScalar = df

                sdir_this = f"{sdir}/session_{idx_session}"
                os.makedirs(sdir_this, exist_ok=True)
                if self.Params["which_level"] in ["trial", "flex"]:
                    return sp._modulationgood_plot_drawings_variables_bytrial(var, vars_conjuction, sdir_this, nplot)
                else:
                    return sp._modulationgood_plot_drawings_variables(var, vars_conjuction, sdir_this, nplot)

    def _modulationgood_plot_drawings_variables_bytrial(self, var, vars_conjuction, sdir,
            nplot = 40):
        """ Plot example drawings across all levels for a given sublevel_conjucntion (figures)
        """

        site = self.Sites[1] # pick any site.
        _, dict_dfs, _ = self.dataextract_as_df_conjunction_vars(var, vars_conjuction, site)

        for lev_others, dfsub in dict_dfs.items():
            print("Plotting .. ", lev_others)
            
            trialcodes = sorted(dfsub["trialcode"].unique().tolist())

            # pick n random
            if len(trialcodes)>nplot:
                import random
                indsplot = sorted(random.sample(range(len(trialcodes)), nplot))
            else:
                indsplot = range(len(trialcodes))

            # Get data
            # WSRONG - is noit matching triaclodes, which are sorted(unique())
            # tmp = dfsub[var].to_list()
            # titles = [tmp[i] for i in indsplot]
            tcs = [trialcodes[i] for i in indsplot]
            
            # convert trialcodes to dataset indices
            inds_dat_beh = [self.SN.datasetbeh_trialcode_to_datidx(tc) for tc in tcs]
            if var in self.SN.Datasetbeh.Dat.columns:
                # title each subplot by its var 
                titles = self.SN.Datasetbeh.Dat.iloc[inds_dat_beh][var].tolist()
            else:
                # Usually becuase this var is realted to something like eye fixations, which are not trial-level.
                titles = self.SN.Datasetbeh.Dat.iloc[inds_dat_beh]["trialcode"].tolist()

            if len(inds_dat_beh)==0 or any([x is None for x in inds_dat_beh]):
                print("trialcodes: ", trialcodes)
                print("tcs: ", tcs)
                print("len(dfsub):", len(dfsub))
                print(inds_dat_beh)
                assert False

            # -- PLOT BEH            
            fig, axes, _ = self.SN.Datasetbeh.plotMultTrials2(inds_dat_beh, "strokes_beh", titles=titles)
            for ax, tit, tcthis in zip(axes.flatten(), titles, tcs):
                ax.set_ylabel(f"{tcthis}")
            fig.savefig(f"{sdir}/lev_others-{'-'.join([str(x) for x in lev_others])}-BEH.pdf")
            
            # -- PLOT TASK            
            fig, axes, _ = self.SN.Datasetbeh.plotMultTrials2(inds_dat_beh, "strokes_task", titles=titles)
            for ax, tit, tcthis in zip(axes.flatten(), titles, tcs):
                ax.set_ylabel(f"{tcthis}")
            fig.savefig(f"{sdir}/lev_others-{'-'.join([str(x) for x in lev_others])}-TASK.pdf")

            plt.close("all")

    def _modulationgood_plot_drawings_variables(self, var, vars_conjuction, sdir,
            nplot = 40):
        """
        Plot example drawings across lewvels for a given variable, seaprate
        figure for each level of the conjucntion variaable.
        """ 

        site = self.Sites[1] # pick any site.
        _, dict_dfs, _ = self.dataextract_as_df_conjunction_vars(var, vars_conjuction, site)

        for lev_others, dfsub in dict_dfs.items():
            print(lev_others)
            index_DS = dfsub["index_DS"].tolist()
            trialcodes = dfsub["trialcode"].tolist()

            # Restrict to trialcodes that exist in dfscalar

            # pick n random
            if len(index_DS)>nplot:
                import random
                indsplot = sorted(random.sample(range(len(index_DS)), nplot))
            else:
                indsplot = range(len(index_DS))

            tmp = dfsub[var].to_list()
            titles = [tmp[i] for i in indsplot]
            inds = [index_DS[i] for i in indsplot]
            tcs = [trialcodes[i] for i in indsplot]

            fig, axes, inds_trials_dataset = self.DS.plot_multiple_overlay_entire_trial(inds, 8);

            for ax, tit, tcthis in zip(axes.flatten(), titles, tcs):
                ax.set_title(f"{var}:{tit}")
                ax.set_ylabel(f"{tcthis}")

            fig.savefig(f"{sdir}/lev_others-{'-'.join([str(x) for x in lev_others])}")

            plt.close("all")


    def modulationgood_plot_list_conjuctions(self, vars_check, SAVEDIR, PLOT = False):
        """ Useful before spending long time on plots, see sample size for different 
        conjunctions of variables. NOTE: picks the first channel and event as arbitrary ones, to
        make sure the output n is accurate (i.e, n trials for a given site and event).
        NOTE: this should get EVERY trial, even for cases where not enough data for spnaning
        multiple levels of a var.
        PARAMS:
        - PLOT, bool, if make plots of all conjucntionsa nd N. This can take a while, if 
        many variables, so better to just print (always does).
        """
        from pythonlib.tools.listtools import partition, powerset
        import os

        sdir = f"{SAVEDIR}/snippets_check_conjunctions_variables"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        
        for var in vars_check:
            assert var in self.DfScalar.columns, f"this var doesnt exist!! {var}"
        if PLOT:
            assert len(vars_check)<7, "are you sure? maybe limit the length of subsets. will take long time."

        site = self.Sites[0] # assume all sites have same trials.
        event = self.DfScalar["event"].unique().tolist()[0]
        RES = {}
        for var in vars_check:
            vars_check_others = [v for v in vars_check if not v==var]
            levels_var = self.DfScalar[var].unique().tolist()
            res = []
            lines = []
            for x in powerset(vars_check_others):
                if len(x)>0:
                    vars_others = list(x)
                    print("var -- vars_others: ", var, ' --- ', vars_others)
                        
                    _, dict_lev_df, _ = self.dataextract_as_df_conjunction_vars(var, vars_others, 
                        site, event, OVERWRITE_n_min=0, OVERWRITE_lenient_n=1)
                    
                    # how many levels of vars_others have enough data?
                    # n_found = len(dict_lev_df)
                    RES[(var, tuple(vars_others))] = len(dict_lev_df)
                    
                    # get n for each lev of others
                    for lev_o, dfthis in dict_lev_df.items():
                        for lev_v in levels_var:
                            n = sum(dfthis[var]==lev_v)
                            res.append({
                                "lev_others":lev_o,
                                "lev_var":lev_v,
                                "vars_others":tuple(vars_others),
                                "n":n
                            })
                            
                            if n < self.ParamsGlobals["n_min_trials_per_level"]:
                                lines.append(f"{tuple(vars_others)} - {lev_o} - {lev_v}  :  [{n}]")
                            else:
                                lines.append(f"{tuple(vars_others)} - {lev_o} - {lev_v}  :  {n}")
                        lines.append(" ")
                    lines.append(" ")                    
            
            if PLOT:
                assert False, "not coded cleanyl, cehck it"
                dftmp = pd.DataFrame(res)
                # Plot for this var
                print("Plotting!! ")
                import seaborn as sns
                fig = sns.catplot(data=dftmp, x="lev_var", y="n", hue="lev_others", kind="point", col="vars_others", col_wrap=4)

                for ax in fig.axes.flatten():
                    ax.axhline(self.ParamsGlobals["n_min_trials_per_level"])    

                fig.savefig(f"{sdir}/{var}_ncounts.pdf")
                plt.close("all")
            
            # Print for this var
        #     from pythonlib.tools.pandastools import grouping_print_n_samples
        #     path = f"{sdir}/ncounts-{var}-vs-varothers-levothers-levvar.txt"
        #     grouping_print_n_samples(dftmp, list_groupouter_grouping_vars=["vars_others", "lev_others", "lev_var"], 
        #                             print_value_not_n=True, savepath=path, save_convert_keys_to_str=True,
        #                             save_as="text")

            from pythonlib.tools.pandastools import grouping_print_n_samples
            path = f"{sdir}/ncounts-{var}-vs-varothers-levothers-levvar.txt"
            from pythonlib.tools.expttools import writeStringsToFile
            writeStringsToFile(path, lines)
            print(path)

        # fig = sns.catplot(data=dftmp, x="lev_var", y="n", hue="lev_others", kind="point", col="vars_others", col_wrap=4)

        # for ax in fig.axes.flatten():
        #     ax.axhline(self.ParamsGlobals["n_min_trials_per_level"])    

        # fig.savefig(f"{sdir}/{var}_ncounts.pdf")
        # plt.close("all")


        # from pythonlib.tools.pandastools import grouping_print_n_samples
        # path = f"{sdir}/ncounts-{var}-vs-varothers-levothers-levvar.txt"
        # grouping_print_n_samples(dftmp, list_groupouter_grouping_vars=["vars_others", "lev_others", "lev_var"], 
        #                          savepath=path, save_convert_keys_to_str=True, save_as="text")



    def modulationgood_plot_summarystats_v2(self, df_var, savedir="/tmp"):
        """
        Takes into account z-score, for significance
        """
        from pythonlib.tools.pandastools import convert_to_2d_dataframe
        from pythonlib.tools.snstools import rotateLabel
        import seaborn as sns

        # Requires this column
        assert "val_zscore" in df_var.columns, "need this for some significance-related analysis"

        # Get aggregation
        dfagg = self.modulationgood_aggregate_df(df_var, aggmethod="weighted_avg") 

        fig = sns.relplot(data=dfagg, x="val_zscore", y="val", col="bregion", col_wrap=4, alpha=0.5)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axvline(0)
            ax.axvline(3, linestyle="--")
        fig.savefig(f"{savedir}/agg_scatter_val_vs_valzscore.pdf")
        plt.close("all")

        fig = sns.catplot(data=dfagg, x="bregion", y="val_zscore", aspect=2, alpha=0.2)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axhline(3)
            
        fig.savefig(f"{savedir}/agg_scatter_valzscore.pdf")


        fig = sns.catplot(data=dfagg, x="bregion", y="val", aspect=2, alpha=0.2)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/agg_scatter_vals.pdf")

        fig = sns.catplot(data=dfagg, x="bregion", y="val", aspect=2, alpha=0.2, kind="bar", ci=68)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/agg_bars_vals.pdf")


        #########
        fig = sns.relplot(data=df_var, x="val_zscore", y="val", col="bregion", col_wrap=4, alpha=0.5)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axvline(0)
            ax.axvline(3, linestyle="--")
        fig.savefig(f"{savedir}/chanxotherlevel_scatter_val_vs_valzscore.pdf")
        plt.close("all")

        fig = sns.catplot(data=df_var, x="bregion", y="val_zscore", aspect=2, alpha=0.2)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axhline(3)
            
        fig.savefig(f"{savedir}/chanxotherlevel_scatter_valzscore.pdf")


        fig = sns.catplot(data=df_var, x="bregion", y="val", aspect=2, alpha=0.2)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/chanxotherlevel_scatter_vals.pdf")

        fig = sns.catplot(data=df_var, x="bregion", y="val", aspect=2, alpha=0.2, kind="bar", ci=68)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/chanxotherlevel_bars_vals.pdf")


        #########
        dfsigs = self.modulationgood_aggregate_df(df_var, aggmethod="n_sig_cases_othervars")

        # Plot
        _, fig, _, _ = convert_to_2d_dataframe(dfsigs, "bregion", "nsig", True);
        fig.savefig(f"{savedir}/nsig_othervarlevels_2d_heat.pdf")

        fig = sns.catplot(data=dfsigs, x="bregion", y="nsig", kind="bar", aspect=2, ci=68)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/nsig_othervarlevels_bar.pdf")


        ##### Print list of sites with strongest modulation
        path = f"{savedir}/dfscalar.csv"
        with open(path, "w") as f:
            df_var.to_csv(f)
            
        path = f"{savedir}/dfscalar_sorted_by_zscore.csv"
        dfout_sorted = df_var.sort_values(by="val_zscore").reset_index(drop=True)
        with open(path, "w") as f:
            dfout_sorted.to_csv(f)
            
        path = f"{savedir}/dfscalar_sorted_by_score.csv"
        dfout_sorted = df_var.sort_values(by="val").reset_index(drop=True)
        with open(path, "w") as f:
            dfout_sorted.to_csv(f)
            

    def modulationgood_plot_twoway_summary(self, df_var, sdir_base):
        """
        Summary plots for 2-way anova reaults.
        PARAMS:
        - df_var, result from modulationgood_compute_wrapper, using a score version
        that is two-way. val_kind column must have values: val_interaction, val_other
        """

        # for each unit, plot pairwise val vs interaction.
        from pythonlib.tools.pandastools import pivot_table
        df_var_pivot = pivot_table(df_var, index=["chan", "var", "var_others", "_event", "event", "val_method", "bregion"], 
                   columns=["val_kind"], values=["val"], flatten_col_names=True)

        # for each unit, use the sum of effects
        def F(x):
            return x["val-val"] + x["val-val_interaction"]
        df_var_mainintercombined = applyFunctionToAllRows(df_var_pivot, F, "val")
        df_var_mainintercombined["val_kind"] = "main_pls_inter"

        ######## 1) Replot
        print("** [TWO-WAY ANOVA] Plotting summarystats")
        sdir = f"{sdir_base}/2anova_modulation_mainplusinter"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        self.modulationgood_plot_summarystats(df_var_mainintercombined, savedir=sdir) 
        plt.close("all")

        ######## 1b) heatmap
        print("** [TWO-WAY ANOVA] Plotting heatmaps")
        sdir = f"{sdir_base}/2anova_modulation_mainplusinter_heatmap"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)
        self.modulationgood_plot_brainschematic(df_var_mainintercombined, sdir) 

        ######## 2) Plot relationship between main and interaction effects
        print("** [TWO-WAY ANOVA] Plotting main-interaction relationships")
        sdir = f"{sdir_base}/2anova_main_inter_rel"
        os.makedirs(sdir, exist_ok=True)
        print(sdir)

        # Relational plot, all same axes.
        fig = sns.relplot(data=df_var_pivot, x="val-val_interaction", y="val-val", col="bregion", row="event")
        for ax in fig.axes.flatten():
            ax.set_xlim(ax.get_ylim())
            ax.set_aspect('equal')
            ax.axhline(0, color="k", alpha=0.3)
            ax.axvline(0, color="k", alpha=0.3)
        fig.savefig(f"{sdir}/relplot-all.pdf")
        plt.close("all")

        # Relational plot, separate for each event (different scales).
        list_event = sorted(df_var_pivot["event"].unique().tolist())
        for ev in list_event:
            dfthis = df_var_pivot[df_var_pivot["event"]==ev]
            fig = sns.relplot(data=dfthis, x="val-val_interaction", y="val-val", col="bregion", col_wrap=4)
            for ax in fig.axes.flatten():
                ax.set_xlim(ax.get_ylim())
                ax.set_aspect('equal')
                ax.axhline(0, color="k", alpha=0.3)
                ax.axvline(0, color="k", alpha=0.3)
            fig.savefig(f"{sdir}/relplot-event_{ev}.pdf")
        plt.close("all")


    def modulationgood_plot_summarystats_fr(self, df_fr, df_fr_levels, savedir="/tmp"):
        """ Plots for analsyis of fr for each (chan, event, var_lev, othervar_lev).
        """
        from neuralmonkey.analyses.anova_agg_plots import _eventwindow_sort
        events_sorted = _eventwindow_sort(df_fr_levels["event"].unique().tolist())

        from pythonlib.tools.snstools import rotateLabel
        fig = sns.catplot(data=df_fr_levels, x="event", hue="vars_others", y="val", col="bregion", kind="point",
                          row ="var", ci=68)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/fr_levels-row_var-1.pdf")

        fig = sns.catplot(data=df_fr_levels, x="event", hue="vars_others", y="val", col="bregion", row ="var", jitter=True, alpha=0.3)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/fr_levels-row_var-2.pdf")


        fig = sns.catplot(data=df_fr_levels, x="event", hue="var", y="val", col="bregion", kind="point",
                          row ="vars_others", ci=68)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/fr_levels-row_varsothers-1.pdf")

        fig = sns.catplot(data=df_fr_levels, x="event", hue="var", y="val", col="bregion", row ="vars_others", jitter=True, alpha=0.3)
        rotateLabel(fig)
        fig.savefig(f"{savedir}/fr_levels-row_varsothers-2.pdf")

        # if df_fr is not None:
        #     try:
        #         fig = sns.catplot(data=df_fr, x="event", y="val", hue="val_kind",
        #                           col="bregion", col_wrap=4, kind="point", aspect=1, height=3,
        #                           order=events_sorted);
        #         rotateLabel(fig)
        #         fig.savefig(f"{savedir}/5_lines_fr.pdf")

        #         fig = plotgood_lineplot(df_fr, xval="event", yval="val", line_grouping="chan",
        #                                 include_mean=True, colvar="bregion", col_wrap=4);
        #         rotateLabel(fig)
        #         fig.savefig(f"{savedir}/5_lineschans_fr.pdf")
        #     except ValueError as err:
        #         pass

        # if df_fr_each_lev is not None:
        #     try:
        #         ################## FR MODULATION BY LEVELS
        #         fig = sns.catplot(data=df_fr_each_lev, x=var, hue="event", y="val", col="bregion", col_wrap=4, kind="point")
        #         rotateLabel(fig)
        #         fig.savefig(f"{savedir}/9_{var}-lines_fr_vs_level.pdf")

        #         fig = plotgood_lineplot(df_fr_each_lev, xval=var, yval="val", line_grouping="chan",
        #                                 include_mean=True, 
        #                                 relplot_kw={"row":"event", "col":"bregion"});
        #         rotateLabel(fig)
        #         fig.savefig(f"{savedir}/9_{var}-lineschans_fr_vs_level.pdf")
        #     except Exception as err:
        #         pass
        plt.close("all")

    def modulationgood_plot_summarystats_groupvars(self, df_var, vars_group, savedir="/tmp"):
        """ Plots, separating into levels of conjuctions of columsn invars_group.
        PARAMS:
        - vars_group, list of str, for grouping --> eacha  single subplot.
        """
        from pythonlib.tools.pandastools import aggregGeneral, append_col_with_grp_index, append_col_with_index_number_in_group
        from pythonlib.tools.pandastools import append_col_with_index_of_level 
        from pythonlib.tools.expttools import writeDictToYaml, writeDictToTxt
        from pythonlib.tools.snstools import rotateLabel
        import seaborn as sns

        from neuralmonkey.analyses.anova_agg_plots import _eventwindow_sort
        events_sorted = _eventwindow_sort(df_var["event"].unique().tolist())

        # 1) Get conjucntions, calling it "dummy"
        df_var = append_col_with_grp_index(df_var, vars_group, "dummy", use_strings=False)
        
        # convert level names to indices, they are too long.
        map_idx_to_level, map_level_to_idx = append_col_with_index_of_level(df_var, "dummy", "dummy_idx")

        # aggregate, to marginalize over other vars.
        df_var_agg = aggregGeneral(df_var, group=["chan", "event", "val_kind", "val_method", "dummy_idx"], 
                      values = ["val"], 
                      nonnumercols=["bregion"])

        #### PLOTS
        var_other_single = "dummy_idx"
        fig = sns.catplot(data=df_var_agg, x="bregion", y="val", row="event", 
            col=var_other_single, kind="bar", ci=68, row_order=events_sorted)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/10_{var_other_single}-1.pdf")
        plt.close("all")

        fig = sns.catplot(data=df_var_agg, x=var_other_single, y="val", hue="event", 
            col="bregion", col_wrap=4, kind="bar", ci=68)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/10_{var_other_single}-2.pdf")
        plt.close("all")

        fig = sns.catplot(data=df_var_agg, x="event", y="val", hue=var_other_single, 
            col="bregion", kind="bar", ci=68, order=events_sorted)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/10_{var_other_single}-3.pdf")
        plt.close("all")

        for sharey in [True, "row"]:
            fig = sns.catplot(data=df_var_agg, x="event", y="val", hue="val_kind",
                              col="bregion", row=var_other_single, aspect=1, height=3, jitter=True,
                             alpha=0.25, sharey=sharey, order=events_sorted);
            for ax in fig.axes.flatten():
                ax.axhline(0)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/10_{var_other_single}-sharey_{sharey}_4.pdf")
            plt.close("all")

        for sharey in [True, "row"]:
            fig = sns.catplot(data=df_var_agg, x="event", y="val", hue="val_kind",
                              col="bregion", row=var_other_single, aspect=1, height=3, kind="bar",
                              ci=68, sharey=sharey, order=events_sorted);
            rotateLabel(fig)
            savefig(fig, f"{savedir}/10_{var_other_single}-sharey_{sharey}_5.pdf")
            plt.close("all")

        # SAve the map from idx to level
        path = f"{savedir}/map_idx_to_level.yaml"
        writeDictToYaml(map_idx_to_level, path)
        path = f"{savedir}/map_idx_to_level.txt"
        writeDictToTxt(map_idx_to_level, path)

    def modulationgood_plot_summarystats(self, df_var, df_fr_each_lev=None, 
            df_fr=None, savedir="/tmp", skip_agg=False):
        """ 
        [GOOD] Plot modulation of fr by a single variable. Uses NEW_VERSION. Pass in dataframes directly.
        - Plots are across all events.
        PARAMS:
        - df_var, dataframe, each row is a (chan, event, variable, level_other_var). with
        modulation by that variable. DO NOT aggregate before passing in.
        REturns from self.modulationgood_compute_wrapper()
        EG:
            chan    var     var_others  lev_in_var_others   event   val_kind    val_method  val     bregion
            0   2   shape_oriented  [gridloc, chunk_within_rank]    ((-1, 1), 0)    00_stroke   modulation_subgroups    r2smfr_minshuff     0.319396    M1_m
            1   2   shape_oriented  [gridloc, chunk_within_rank]    ((1, 1), 1)     00_stroke   modulation_subgroups    r2smfr_minshuff     0.188035    M1_m
            2   2   shape_oriented  [gridloc, chunk_within_rank]    ((0, 0), 0)     00_stroke   modulation_subgroups    r2smfr_minshuff     0.401434    M1_m
            3   15  shape_oriented  [gridloc, chunk_within_rank]    ((-1, 1), 0)    00_stroke   modulation_subgroups    r2smfr_minshuff     0.032230    M1_m
            4   15  shape_oriented  [gridloc, chunk_within_rank]    ((1, 1), 1)     00_stroke   modulation_subgroups    r2smfr_minshuff     0.089292    M1_m
        - var, string, which variable for plots to care about [in progress for df_var]
        - df_fr_each_lev, holding fr for each (chan, event, variable, level_variable)
        - df_fr, holding fr for each (chan, event)
        For the latter two, hacky code:
            from pythonlib.tools.pandastools import aggregGeneral
            df_fr = aggregGeneral(SP.DfScalar, group=["chan", "event_aligned"], values = ["fr_scalar_raw"])
            # give bregion
            df_fr = SP.datamod_append_bregion(df_fr)
            df_fr["event"] = df_fr["event_aligned"]

            df_fr["val"] = df_fr["fr_scalar_raw"]
            df_fr["val_kind"] = "fr_scalar_raw"


            from pythonlib.tools.pandastools import aggregGeneral
            df_fr_levels = aggregGeneral(SP.DfScalar, group=["chan", "event_aligned", var], values = ["fr_scalar"])
            # give bregion
            df_fr_levels = SP.datamod_append_bregion(df_fr_levels)
            df_fr_levels["event"] = df_fr_levels["event_aligned"]

            df_fr_levels["val"] = df_fr_levels["fr_scalar"]
            df_fr_levels["val_kind"] = "fr_each_level"
            df_fr_levels
        """
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.snstools import plotgood_lineplot
        import seaborn as sns
        from pythonlib.tools.pandastools import aggregGeneral

        df_var = df_var.reset_index(drop=True)

        list_var = sort_mixed_type(df_var["var"].unique().tolist())
        list_var_others = sort_mixed_type(df_var["var_others"].unique().tolist())
        assert len(list_var)==1, "not yet coded! shoudl use this as lower loevel code, iterate over vars outside"
        assert len(list_var_others)==1
        print("Saving at:", savedir)
        var = list_var[0]
        var_others = list_var_others[0] 
        print("Found this var: ", var)
        print("Found this var_others: ", var_others)

        # Aggreagated
        if skip_agg:
            df_var_chan = df_var.copy()
        else:
            print("Aggregating dataframe over all othervars ...")
            if "n_datapts" in df_var.columns:
                # Then this is one-way, has n datapts for each othervar_level:
                df_var_chan = self.modulationgood_aggregate_df(df_var)
            else:
                # then this is probably 2-way, dont even need to aggregate...
                df_var_chan = aggregGeneral(df_var, group=["chan", "event", "var", "var_others", "val_kind", "val_method"], values = ["val"], 
                  nonnumercols=["bregion"]) # one datapt per chan

        # order the events
        # events_sorted = sorted(df_var_chan["event"].unique())
        from neuralmonkey.analyses.anova_agg_plots import _eventwindow_sort
        events_sorted = _eventwindow_sort(df_var_chan["event"].unique().tolist())

        print("Plotting ...")
        # Compare across events
        fig = sns.catplot(data=df_var_chan, x="event", y="val", hue="val_kind",
                          col="bregion", row="var", kind="point", ci=68, aspect=1, height=5,
                          order=events_sorted);
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k")
        rotateLabel(fig)
        fig.savefig(f"{savedir}/1_lines_modulation_kinds.pdf")

        fig = sns.catplot(data=df_var_chan, x="event", y="val", hue="val_kind",
                          col="bregion", row="var", aspect=1, height=5, jitter=True, 
                          alpha=0.25, order=events_sorted);
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k")
        fig.savefig(f"{savedir}/1_scatter_modulation_kinds.pdf")

        # Comparing brain regions
        fig = sns.catplot(data=df_var_chan, x="bregion", y="val", col="event", hue="val_kind",
                    row="var", kind="bar", ci=68, aspect=2, height=4, col_order=events_sorted);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/2_bars_feature_vs_event.pdf")      

        fig = sns.catplot(data=df_var_chan, x="bregion", y="val", col="event", hue="val_kind",
                    row="var", aspect=2, height=4, alpha=0.25, col_order=events_sorted);
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k")        
        fig.savefig(f"{savedir}/2_scatter_feature_vs_event.pdf")      

        # Comparing brain regions
        if False:
            fig = sns.catplot(data=df_var_chan, x="bregion", y="val", col="val_kind",
                        row="event", kind="bar", ci=68, hue="var", aspect=2, height=4,
                        row_order=events_sorted);
            rotateLabel(fig)
            fig.savefig(f"{savedir}/2_bars_feature_vs_valkind.pdf")

        if False:
            # Is usually identical to 1_...
            # Compare across events
            fig = sns.catplot(data=df_var_chan, x="event", y="val", hue="var",
                              col="bregion", row="val_kind", kind="point", ci=68, aspect=1, 
                              height=5, order=events_sorted);
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k")        
            fig.savefig(f"{savedir}/3_lines_valkinds.pdf")

        from pythonlib.tools.snstools import plotgood_lineplot
        for val_kind in df_var_chan["val_kind"].unique():
            df_var_chan_this = df_var_chan[df_var_chan["val_kind"]==val_kind]
            fig = plotgood_lineplot(df_var_chan_this, xval="event", yval="val", line_grouping="chan",
                                    include_mean=True, 
                                    relplot_kw={"row":"var", "col":"bregion"});
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k")        
            fig.savefig(f"{savedir}/4_lineschans_feature_vs_region.pdf")
        plt.close("all")

        # Comparing brain regions
        fig = sns.catplot(data=df_var_chan, x="bregion", y="val", col="val_kind",
                    row="var", kind="bar", ci=68, hue="event", aspect=3, height=4);
        rotateLabel(fig)
        fig.savefig(f"{savedir}/7_bars_feature_vs_valkind.pdf")


        ####### MODULATION FOR EACH LEVEL_OTHER, AND GROUPINGS OF LEVEL_OTHER.
        # plot summary, grouping by each marginal level of vars_others
        if "lev_in_var_others" in df_var.keys() and len(df_var["lev_in_var_others"].unique().tolist())>1: # then is one-way anova, and has >1 level for the 2nd var.
        # if len(df_var["lev_in_var_others"].unique().tolist())>1:\

            if len(df_var["lev_in_var_others"].unique().tolist())<6:
                try:
                    fig = sns.catplot(data=df_var, x="bregion", y="val", row="event", 
                        col="lev_in_var_others", kind="bar", ci=68, row_order=events_sorted)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/11_lev_in_var_others-1.pdf")
                    plt.close("all")

                    fig = sns.catplot(data=df_var, x="lev_in_var_others", y="val", hue="event", 
                        col="bregion", col_wrap=4, kind="bar", ci=68)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/11_lev_in_var_others-2.pdf")
                    plt.close("all")

                    fig = sns.catplot(data=df_var, x="event", y="val", hue="lev_in_var_others", 
                        col="bregion", kind="bar", ci=68, order=events_sorted)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/11_lev_in_var_others-3.pdf")
                    plt.close("all")
                except Exception as err:
                    pass

            # Plot for each level of user-defined conjucntions
            from neuralmonkey.metadat.analy.anova_params import params_getter_plots, LIST_ANALYSES
            if hasattr(self, "ParamsDfvar") and "ANALY_VER" in self.ParamsDfvar and self.ParamsDfvar["ANALY_VER"] in LIST_ANALYSES:
                sn = self._session_extract_all()[0]
                params = params_getter_plots(sn.Animal, sn.Date, self.Params["which_level"], self.ParamsDfvar["ANALY_VER"])
                VARS_GROUP_PLOT = params["VARS_GROUP_PLOT"]
                for vars_group in VARS_GROUP_PLOT:
                    savedirthis = f"{savedir}/vars_group/{'-'.join(vars_group)}"
                    os.makedirs(savedirthis, exist_ok=True)
                    if all([v in df_var.columns for v in vars_group]):
                        print("Saving plots at: ", savedirthis)
                        self.modulationgood_plot_summarystats_groupvars(df_var, vars_group, savedirthis)

            for var_other_single in var_others: # var_other_single = "stroke_index"

                if var_other_single=="DUMMY":
                    print(f"SKIPPING plotting for specific single other var, since it is called DUMMY")
                    continue

                try:
                    print(f"Plotting for specific single other var: {var_other_single}...")
                    # Do need to do this, since it marginalizes over the other variables
                    df_var_agg = aggregGeneral(df_var, group=["chan", "event", "var", "var_others", "val_kind", "val_method", var_other_single], 
                                  values = ["val"], 
                                  nonnumercols=["bregion"])

                    # fig = sns.catplot(data=df_var_agg, x=var_other_single, y="val", row="event", col="bregion", kind="bar")
                    # rotateLabel(fig)
                    # fig.savefig(f"{savedir}/10_bars_vs_marginalvarother_{var_other_single}-1.pdf")

                    fig = sns.catplot(data=df_var_agg, x="bregion", y="val", row="event", 
                        col=var_other_single, kind="bar", ci=68, row_order=events_sorted)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/10_{var_other_single}-1.pdf")

                    # fig = sns.catplot(data=df_var, x="bregion", y="val", hue="event", col=var_other_single, col_wrap=4, kind="bar")
                    # rotateLabel(fig)
                    # fig.savefig(f"{savedir}/10_bars_vs_each_level_varother-1.pdf")

                    fig = sns.catplot(data=df_var_agg, x=var_other_single, y="val", hue="event", 
                        col="bregion", col_wrap=4, kind="bar", ci=68)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/10_{var_other_single}-2.pdf")

                    fig = sns.catplot(data=df_var_agg, x="event", y="val", hue=var_other_single, 
                        col="bregion", kind="bar", ci=68, order=events_sorted)
                    rotateLabel(fig)
                    fig.savefig(f"{savedir}/10_{var_other_single}-3.pdf")

                    plt.close("all")

                    for sharey in [True, "row"]:
                        fig = sns.catplot(data=df_var_agg, x="event", y="val", hue="val_kind",
                                          col="bregion", row=var_other_single, aspect=1, height=3, jitter=True,
                                         alpha=0.25, sharey=sharey, order=events_sorted);
                        for ax in fig.axes.flatten():
                            ax.axhline(0)
                        rotateLabel(fig)
                        fig.savefig(f"{savedir}/10_{var_other_single}-sharey_{sharey}_4.pdf")


                    for sharey in [True, "row"]:
                        fig = sns.catplot(data=df_var_agg, x="event", y="val", hue="val_kind",
                                          col="bregion", row=var_other_single, aspect=1, height=3, kind="bar",
                                          ci=68, sharey=sharey, order=events_sorted);
                        rotateLabel(fig)
                        fig.savefig(f"{savedir}/10_{var_other_single}-sharey_{sharey}_5.pdf")

                    plt.close("all")
                except ValueError as err:
                    # NOT SURE WHYA..

                    # Traceback (most recent call last):
                    #   File "analy_anova_plot.py", line 305, in <module>
                    #     SP.modulationgood_compute_plot_ALL(var, vars_conjuction, 
                    #   File "/gorilla1/code/neuralmonkey/neuralmonkey/classes/snippets.py", line 1575, in modulationgood_compute_plot_ALL
                    #     self.modulationgood_plot_summarystats(df_var, df_fr_levels, df_fr, savedir=sdir) 
                    #   File "/gorilla1/code/neuralmonkey/neuralmonkey/classes/snippets.py", line 3652, in modulationgood_plot_summarystats
                    #     fig = sns.catplot(data=df_var_agg, x="event", y="val", hue=var_other_single, 
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/_decorators.py", line 46, in inner_f
                    #     return f(**kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/categorical.py", line 3847, in catplot
                    #     g.map_dataframe(plot_func, x=x, y=y, hue=hue, **plot_kws)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/axisgrid.py", line 777, in map_dataframe
                    #     self._facet_plot(func, ax, args, kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/axisgrid.py", line 806, in _facet_plot
                    #     func(*plot_args, **plot_kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/_decorators.py", line 46, in inner_f
                    #     return f(**kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/categorical.py", line 3190, in barplot
                    #     plotter.plot(ax, kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/categorical.py", line 1639, in plot
                    #     self.draw_bars(ax, bar_kws)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/seaborn/categorical.py", line 1622, in draw_bars
                    #     barfunc(offpos, self.statistic[:, j], self.nested_width,
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/matplotlib/__init__.py", line 1442, in inner
                    #     return func(ax, *map(sanitize_sequence, args), **kwargs)
                    #   File "/home/lucast4/miniconda3/envs/drag2_matlab/lib/python3.8/site-packages/matplotlib/axes/_axes.py", line 2436, in bar
                    #     raise ValueError(f'number of labels ({len(patch_labels)}) '
                    # ValueError: number of labels (2) does not match number of bars (9).

                    pass

        # # AVGMOD
        # dfthis = dfdat_var_methods
        # fig = sns.catplot(data=dfthis, x="val_kindmethod", y="val", hue="var",
        #                   col="bregion", col_wrap=4, kind="point", aspect=1, height=3);
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/6_lines_avgmodulation_vs_method.pdf")

        # # AVGMOD
        # dfthis = dfdat_var_methods
        # fig = plotgood_lineplot(dfthis, xval="method", yval="val", line_grouping="chan",
        #                         include_mean=True, 
        #                         relplot_kw={"row":"var", "col":"bregion"});
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/6_lineschans_avgmodulation_vs_method.pdf")


        # dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_delt" for var in list_var])]
        # # fig = sns.catplot(data=dfthis, x="val_kind", y="val", col="bregion", col_wrap=4, aspect=1.5, height=2,
        # #                  kind="bar");
        # fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind", aspect=2, height=4, kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_deltsonly.pdf")

        # dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_mean" for var in list_var])]
        # fig = sns.catplot(data=dfthis, x="bregion", y="val", col="val_kind", aspect=2, height=4, kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_meansonly.pdf")


        # dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_delt" for var in list_var])]
        # fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
        #                  kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_features_deltonly.pdf")

        # dfthis = dfdat_summary_mod[dfdat_summary_mod["val_kind"].isin([f"{var}_mean" for var in list_var])]
        # fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
        #                  kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_features_meanonly.pdf")


        # dfthis = dfdat_summary_mod
        # fig = sns.catplot(data=dfthis, x="val_kind", y="val", col="bregion", col_wrap=4, aspect=1.5, height=2,
        #                  kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_all.pdf")
        # # fig = sns.catplot(data=dfthis, x="bregion", y="val", hue="val_kind",aspect=2, height=4,
        # #                  kind="bar");
        # # from pythonlib.tools.snstools import rotateLabel
        # # rotateLabel(fig)

        # fig = sns.catplot(data=dfthis, x="val_kind", y="val", hue="bregion",aspect=3, height=4,
        #                  kind="bar");
        # rotateLabel(fig)
        # fig.savefig(f"{savedir}/8_bars_features_all.pdf")


    def modulationgood_plot_brainschematic(self, df_var, sdir):
        """
        Plot heatmaps, each subplot is an event, showing mean score for each
        (bregion, event), aggregated from df_var
        """
        ######## HEATMAP (brain schematic)
        from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform

        for valkind in df_var["val_kind"].unique().tolist():
            dfthis = df_var[df_var["val_kind"]==valkind]

            # Plot all, including movement
            plot_df_from_longform(dfthis, "val", "event", savedir=sdir, savesuffix=f"{valkind}")

            # remove movement event
            inds_motor = dfthis["_event"].str.contains("stroke")
            dfthismotor = dfthis[~inds_motor]
            if len(dfthismotor)>0:
                plot_df_from_longform(dfthismotor, "val", "event", savedir=sdir, savesuffix=f"{valkind}-NOMOTOR")

            plt.close("all")

    ########################################## STATE SPACE STUFF
    # def _statespace_pca_extract_data(self, sites, event, pre_dur, post_dur,
    #         var, list_vars_others, do_balance, pre_dur, post_dur):
    #     # 1) Extract data 
    #     dfthis = self.dataextract_as_df_multsites_wrapper(sites, event, 
    #         var, list_vars_others, do_balance, pre_dur, post_dur)

    #     # if do_balance:
    #     #     assert isinstance(balance_var, str)
    #     #     assert isinstance(list_balance_vars_others, list)

    #     #     #### FIRST, get pruned dataset for a single site.
    #     #     # Then no missing conjunctions of levels of var and vars_others
    #     #     Sites = dfthis["chan"].unique().tolist()
    #     #     # 2) Extract, pruning by sample sizes of conjunctions.
    #     #     # just pick the first site, since now all sites have exacly the same trials.
    #     #     _dfthis_single, _, _ = self.dataextract_as_df_conjunction_vars(balance_var, list_balance_vars_others, 
    #     #                                     site=Sites[0], DFTHIS=dfthis, 
    #     #                                     DEBUG_CONJUNCTIONS=False, balance_no_missed_conjunctions=True)

    #     #     #### SECOND, use result from single site to filter all data
    #     #     # dfthis tells you which datinds to keep to still have good n data
    #     #     # Apply this filter to keep only these datinds for all sites.
    #     #     print("Starting len:", len(dfthis))
    #     #     list_index_datapt = _dfthis_single["index_datapt"].unique().tolist()
    #     #     print("These are the final good index datapts: ")
    #     #     print(list_index_datapt)
    #     #     dfthis = dfthis[dfthis["index_datapt"].isin(list_index_datapt)].reset_index(drop=True)
    #     #     print("Ending len:", len(dfthis))

    #     #     # TODO: balance the n data as well

    #     ##### Now you have fully balanced dataset, get PCA space
    #     # For every trial, get a PA across all chans. then concatenate the PA.
    #     # - first, assign a datapt index 

    #     # Convert to PA
    #     # The reason is simply to use the methods within there for generating PCA space
    #     PA = self._dataextract_as_popanal_good(dfthis)    

    #     return PA

    def statespate_pca_extract_data(self, PApca, event, var=None, vars_others=None,
            levels_var = None, levels_othervar=None):
        """
        Extract neural data for this event, makign sure chans match those in the 
        Popanal holding PC space.
        RETURNS:
        - DICT_DF_DAT, levels_var, levels_othervar, seee within
        """

        # list_event = SP.DfScalar["event"].unique().tolist()
        sites = PApca.Chans
        return self._statespace_pca_extract_data(sites, event, var, vars_others,
            levels_var = levels_var, levels_othervar=levels_othervar)

    def _statespace_pca_extract_data(self, sites, event, var=None, vars_others=None,
            pre_dur=None, post_dur=None, PRINT=False,
            levels_var = None, levels_othervar=None):
        """ Get population data for plotting
        RETURNS:
        DICT_DF_DAT, levels_var, levels_othervar
        -- OR
        - empty dframne, None, None, if not data
        """
        dfdat = self.dataextract_as_df_multsites_wrapper(sites, event, pre_dur=pre_dur, post_dur=post_dur)

        if len(dfdat)==0:
            return dfdat, None, None

        if var is not None:
            # Split into each level
            DICT_DF_DAT, levels_var, levels_othervar = self.dataextract_as_df_split_into_var_conjunctionvar(dfdat, 
                var, vars_others, levels_var, levels_othervar)
            
        else:
            # keep all data. put in a dict
            DICT_DF_DAT = {"alldata":dfdat}
            levels_var = ["alldata"]
            levels_othervar = None

        if PRINT:
            print("each df to plot (diff color):")
            for k, v in DICT_DF_DAT.items():
                print(k, len(v))    

        return DICT_DF_DAT, levels_var, levels_othervar

    def dataextract_as_popanal_statespace(self, sites, event, pre_dur=None, post_dur=None,
                                      do_balance=False, balance_var=None, balance_list_vars_others=None,
                                      exclude_othervar_levels_missing_any_var_level=False,
                                      list_features_extraction=None,
                                      which_fr_sm = "fr_sm", max_frac_trials_lose=None):
        """ [GOOD] Use this for all population-level analyses.
        Extract data that is clean - i.e., (i) balanced to get
        all combo of trials x chans x timepoints in a single PA. (ii) Can also choose to
        balance across variables (e.g., each conjuction of variables must have at least n datapts.
        (iii) prune to specific time window.
        PARAMS;
        - dfthis, slice of self.DfScalar, for a single event.
        - do_balance, bool, if True, then makes sure each conjucjtion of levle of var and vars others has data,
        i.e, then makes sure the resulting dfthis is "square" in that each level of var has at least some data for
        each level of vars_others. Does this in an interative fashion (see inner code).
        - balance_var, str,.
        - balance_list_vars_others, list of str, other vars, to cross with balancevary (ecah conjucntion of this)
        - exclude_othervar_levels_missing_any_var_level, bool, if True, then only keeps levels of othervar which have at least one datapt for each level of var
        RETURNS:
            - PA, single (chans, trials, times)
            - sample_meta, dict, metaparams for subsampling, balancing etc

        """
        from pythonlib.tools.pandastools import grouping_print_n_samples

        if do_balance:
            assert isinstance(balance_var, str)
            assert isinstance(balance_list_vars_others, list)

        # 1) Extract data
        dfthis = self.dataextract_as_df_multsites_wrapper(sites, event,
            balance_var, balance_list_vars_others, do_balance, pre_dur, post_dur,
            exclude_othervar_levels_missing_any_var_level=exclude_othervar_levels_missing_any_var_level)

        if len(dfthis)==0:
            print(sites)
            print(event)
            print(balance_var)
            print(balance_list_vars_others)
            raise NotEnoughDataException

        # save the num conjunctions
        if do_balance:
            if balance_list_vars_others is not None:
                grpdict = grouping_print_n_samples(dfthis, balance_list_vars_others+[balance_var])
            else:
                grpdict = grouping_print_n_samples(dfthis, [balance_var])

            sample_meta = {
                "grpdict":grpdict
            }
        else:
            sample_meta = {}

        # TODO: balance the n data as well

        # Convert to PA
        # The reason is simply to use the methods within there for generating PCA space
        PA = self.dataextract_as_popanal_good(dfthis, list_features_extraction=list_features_extraction,
                                    which_fr_sm = which_fr_sm, chans_needed=sites,
                                              max_frac_trials_lose=max_frac_trials_lose)

        for var in list_features_extraction:
            assert var in PA.Xlabels["trials"].columns
        return PA, sample_meta


    def dataextract_as_popanal_statespace_balanced_pca(self, sites, event, pre_dur, post_dur,
                                                       var=None, list_vars_others=None, do_balance=False,
                                                       pca_trial_agg_grouping=None, pca_time_agg_method=None,
                                                       pca_norm_subtract_condition_invariant=False,
                                                       pca_plot=True,
                                                       exclude_othervar_levels_missing_any_var_level=True):
        """
        Extract space that is perfectly balanced, so that each variable conjunction has at least one
        datpt for each level of var, and vice versa (ie.., square), and optionally return a processed version after applying
        PCA (dim=chans, data=trials, or trial_means after grouping).
        PARAMS;
        - dfthis, slice of self.DfScalar, for a single event. 
        - do_balance, bool, if True, then makes sure each conjucjtion of levle of var and vars others has data
        - balance_var, str
        - list_balance_vars_others, list of str, other vars, to cross with balancevary (ecah conjucntion of this)
        - pca_trial_agg_grouping, list of str or None, mean values for this group will be taken before computing pca. e..g,
        if ["seqc_0_shape"], then pca is done on mean values, one for each shape.
        - pca_time_agg_method, either None, or ",mean" where mean takes mean over time
        - trial_var, name of column which counts as a single datapt. is used for extracting balanced data.
        """
        from neuralmonkey.analyses.state_space_good import pca_make_space

        assert list_vars_others is None, "not yet coded!!!"
        if var is None and pca_trial_agg_grouping is not None:
            # you should prune.
            var = pca_trial_agg_grouping

        PA, sample_meta = self.dataextract_as_popanal_statespace(sites, event, pre_dur, post_dur,
                                      do_balance, var, list_vars_others,
                                      exclude_othervar_levels_missing_any_var_level)

        #### DO PCA
        if pca_trial_agg_grouping is None:
            trial_agg_method = None
        else:
            trial_agg_method = "grouptrials"

        PApca, fig = PA.pca_make_space(trial_agg_method=trial_agg_method, trial_agg_grouping=pca_trial_agg_grouping,
                       time_agg_method=pca_time_agg_method, 
                       norm_subtract_condition_invariant=pca_norm_subtract_condition_invariant,
                       ploton=pca_plot)

        return PApca, fig, PA, sample_meta

    ################################################
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
    # def plotwrapper_heatmap_smfr(self, which_var = "event_aligned", sdir=None):
    #     """ Plot smoothed, average FR in heatmap, one figure for each region,
    #     one subplot for each event, and each unit a row in this subplot
    #     """

    #     sn = self.SN
    #     list_event_aligned = self.DfScalar[which_var].unique().tolist()

    #     if which_var!="event_aligned":
    #         event0 = self.Params["list_events_uniqnames"][0]
    #         assert len(self.Params["list_events_uniqnames"])==1, "multipoel events ,not sure what to use for timing..."

    #     # if which_var=="event_aligned":
    #     #     list_event_aligned = self.Params["list_events_uniqnames"]
    #     # else:
    #     #     list_event_aligned = self.Params["map_var_to_levels"][which_var]
    #     #     event0 = self.Params["list_events_uniqnames"][0]
    #     #     assert len(self.Params["list_events_uniqnames"])==1, "multipoel events ,not sure what to use for timing..."


    #     ZSCORE = True
    #     list_regions = sn.sitegetter_get_brainregion_list()
    #     zlims = [-2, 2]
    #     # zlims = [None, None]

    #     for region in list_regions:
    #         print("Plotting...", region)
    #         sites_this = [s for s in self.Sites if s in sn.sitegetter_map_region_to_sites(region)]

    #         # 1) extract smoothed FR for each  unit, for each event
    #         fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*4, 8))
            
    #         # 2) Collect all fr mat
    #         List_fr_mat = []
    #         for i, (event, ax) in enumerate(zip(list_event_aligned, axes.flatten())):
                            
    #             # extract matrix of mean fr (site x time)
    #             list_frmean = []
    #             for site in sites_this:
    #                 frmat, frmat_times, dict_plot_vals = self.dataextract_as_frmat(site, var=which_var, 
    #                     var_level=event, return_as_zscore=ZSCORE) 
    #                 frmean = np.mean(frmat, 0)
    #                 list_frmean.append(frmean)

    #             frmat_site_by_time = np.stack(list_frmean)

    #             # sort (only for first index)
    #             if i==0:
    #                 frmat_site_by_time, sites_this = self.frmat_sort_by_time_of_max_fr(frmat_site_by_time, 
    #                     sites_this)

    #             # normalize firing rates (use percent change from first time bin)
    #             if False:
    #                 frmedian = np.median(frmat_site_by_time, axis=1, keepdims=True)
    #                 frmat_site_by_time = (frmat_site_by_time - frmedian)/frmedian

    #             List_fr_mat.append(frmat_site_by_time)

    #         # Figure out zlim
    #         zmax = np.max(np.abs([x.max() for x in List_fr_mat] + [x.min() for x in List_fr_mat]))
    #         if zmax>1.8:
    #             zmax = 1.8
            
    #         for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
    #             # if which_var=="event_aligned":
    #             #     times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event)
    #             # else:
    #             #     times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event0)

    #             # plot as 2d heat map
    #             from pythonlib.tools.snstools import heatmap_mat
    #         #         sns.heatmap(frmat_site_by_time, ax=ax, )
    #             _, ax, _ = heatmap_mat(frmat_site_by_time, annotate_heatmap=False, ax=ax, diverge=True, zlims=[-zmax, zmax]);

    #             # Set y and x ticks
    #             ax.set_yticks([i+0.5 for i in range(len(sites_this))], labels=sites_this)
    #             ax.set_xticks(dict_plot_vals["xticks"], labels=dict_plot_vals["xtick_labels"])
    #             ax.axvline(dict_plot_vals["ind_aligned_to_event"])
    #             ax.set_title(event)

    #         fig.savefig(f"{sdir}/zscored-{region}.pdf")
            
            
    #         # plot the means
    #         fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*3, 3))
    #         from neuralmonkey.neuralplots.population import plotNeurTimecourse, plot_smoothed_fr

    #         for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
    #             plot_smoothed_fr(frmat_site_by_time, frmat_times, ax=ax)
    #             ax.axvline(dict_plot_vals["ind_aligned_to_event"])
    #             ax.set_title(event)
                    
    #         if sdir:
    #             fig.savefig(f"{sdir}/zscored-{region}-mean.pdf")
            
    #         assert False
    #         plt.close("all")        

    def plotgood_heatmap_smfr(self, which_var = "event_aligned", sdir=None,
        ZSCORE = True):
        """ [Good]
        Plot heatmaps of fr aligned to each event (column) for all units (rows),
        z-scored (to allow comparison across sites) across trials.
        Also plot the mean for each area (same represntation).
        One figure for each region,
        PARAMS;
        - which_var, str, defines the columns

        """
        from neuralmonkey.neuralplots.population import plotNeurTimecourse, plot_smoothed_fr
        from pythonlib.tools.snstools import heatmap_mat

        # if ZSCORE:
        #     zlims = [-2, 2]
        # else:
        #     zlims = [None, None]

        ## Get the events (columns)
        list_event_aligned = self.DfScalar[which_var].unique().tolist()
        if which_var!="event_aligned":
            assert len(self.Params["list_events_uniqnames"])==1, "multipoel events ,not sure what to use for timing..."
            event0 = self.Params["list_events_uniqnames"][0]
        
        ## Iter over each region
        list_regions = self.SN.sitegetter_get_brainregion_list_BASE()
        for region in list_regions:
            print("Plotting...", region)
            sites_this = [s for s in self.Sites if s in self.SN.sitegetter_map_region_to_sites(region)]

            # 1) extract smoothed FR for each  unit, for each event
            fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*4, 8), squeeze=False)
            
            # 2) Collect all fr mat
            List_fr_mat = []
            for i, (event, ax) in enumerate(zip(list_event_aligned, axes.flatten())):
                            
                # extract matrix of mean fr (site x time)
                list_frmean = []
                for site in sites_this:
                    frmat, frmat_times, dict_plot_vals = self.dataextract_as_frmat(site, var=which_var, 
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
            if ZSCORE:
                zmax = np.max(np.abs([x.max() for x in List_fr_mat] + [x.min() for x in List_fr_mat]))
                if zmax>1.8:
                    zmax = 1.8
                zlims = [-zmax, zmax]
            else:
                zlims = [None, None]
            
            ## Plot for each event
            for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
                # if which_var=="event_aligned":
                #     times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event)
                # else:
                #     times, ind_0, xticks, xticklabels = self.event_extract_time_labels(event0)

                # plot as 2d heat map
            #         sns.heatmap(frmat_site_by_time, ax=ax, )
                _, ax, _ = heatmap_mat(frmat_site_by_time, annotate_heatmap=False, ax=ax, diverge=True, zlims=zlims);

                # Set y and x ticks
                ax.set_yticks([i+0.5 for i in range(len(sites_this))], labels=sites_this)
                ax.set_xticks(dict_plot_vals["xticks"], labels=dict_plot_vals["xtick_labels"])
                ax.axvline(dict_plot_vals["ind_aligned_to_event"])
                ax.set_title(event)
            savefig(fig, f"{sdir}/zscored_{ZSCORE}-{region}.pdf")
            
            ## plot the means
            fig, axes = plt.subplots(1, len(list_event_aligned), sharex=False, sharey=True, figsize=(len(list_event_aligned)*3, 3), squeeze=False)

            for event, frmat_site_by_time, ax in zip(list_event_aligned, List_fr_mat, axes.flatten()):
                plot_smoothed_fr(frmat_site_by_time, frmat_times, ax=ax)
                ax.axvline(0)
                ax.set_title(event)
            savefig(fig, f"{sdir}/zscored_{ZSCORE}-{region}-mean.pdf")
            
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

    def _plotgood_smoothfr(self, site, event=None, ax=None):
        """ 
        """

        assert ax is not None, "not yet coded."

        # Extract data
        dfthis = self.dataextract_as_df_good(site, event)

        # Get event boundaries
        tmp = self._plotgood_rasters_extract_xmin_xmax()
        event_bounds = [tmp[0], 0., tmp[1]]

        pathis = self._dataextract_as_popanal_singlesite_OLD(dfthis, [site])
        pathis.plotwrapper_smoothed_fr(ax=ax, plot_indiv=False, plot_summary=True)


    def _plotgood_smoothfr_average_each_level(self, site, var, vars_others=None,
        event=None, plot_these_levels_of_varsothers=None, plot_on_these_axes=None,
                                              leave_subplot_empty_if_no_data=False,
                                              OVERWRITE_n_min=None, OVERWRITE_lenient_n=None,
                                                                                balance_same_levels_across_ovar=False):
        """ Low-level plot, each subplot is a single level of var-Others,
        and within each, plot eachlevel for var.
        Figure is a single level of (site, event)
        PARAMS:
        - plot_these_levels_of_varsothers, list of values, which will each be a subplot.
        also defines the order of subplots. skips any values that dont exist.
        - plot_on_these_axes, list of axes to plot on.
        - leave_subplot_empty_if_no_data, if True, maintains one to one mapping between axes and
        plot_these_levels_of_varsothers, even if a subplot needs to be empty to do so.
        """

        # Extract data
        dict_lev_pa, levels_var = self._dataextract_as_popanal_conjunction_vars_OLD(var,
                                                                                    vars_others, site, event=event,
                                                                                    OVERWRITE_n_min=OVERWRITE_n_min, OVERWRITE_lenient_n=OVERWRITE_lenient_n,
                                                                                balance_same_levels_across_ovar=balance_same_levels_across_ovar)

        # for k, pa in dict_lev_pa.items():
        #     print(k, pa.X.shape, pa.Xlabels["trials"][var].unique())
        # assert False

        # Prune to data you want.
        if plot_these_levels_of_varsothers:
            if leave_subplot_empty_if_no_data:
                dict_lev_pa = {k:(dict_lev_pa[k] if k in dict_lev_pa.keys() else None) for k in plot_these_levels_of_varsothers}
            else:
                # dict_lev_pa = {k:v for k, v in dict_lev_pa.items() if k in plot_these_levels_of_varsothers}
                dict_lev_pa = {k:dict_lev_pa[k] for k in plot_these_levels_of_varsothers if k in dict_lev_pa.keys()}

        # Get event boundaries
        tmp = self._plotgood_rasters_extract_xmin_xmax()
        event_bounds = [tmp[0], 0., tmp[1]]

        # get common levels of var (for assiging each its own color)
        levels_var_exist = []
        for pa in dict_lev_pa.values():
            if pa is not None:
                levels_var_exist.extend(pa.Xlabels["trials"][var].unique().tolist())
        levels_var_exist = sort_mixed_type(set(levels_var_exist))[::-1] # to match the rasters

        # get common y axis.
        if False:
            # was trying to make them sahare axes. but used better way.
            ymaxes = []
            for pa in dict_lev_pa.values():
                tmp = np.mean(pa.X, axis=0, keepdims=True) # mean over chans
                print(tmp.shape)
                tmp = np.mean(tmp, axis=1, keepdims=True) # mean over trials
                print(tmp.shape)
                ymax = np.max(tmp)
                ymaxes.append(ymax)
            print(ymaxes)
            YMAX = max(ymaxes)

        if plot_on_these_axes is None:
            # Generate axes
            ncols = 4
            ngrid = len(dict_lev_pa)
            nrows = int(np.ceil(ngrid/ncols))
            # dur = tmp[1] - tmp[0]
            figsize = (ncols*3, nrows*3)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        else:
            fig = None
            axes = plot_on_these_axes
            assert len(axes)>=len(dict_lev_pa), "not enough axes"

        # joint he axes
        if len(axes.flatten())>0:
            from pythonlib.tools.plottools import share_axes
            share_axes(axes, "y")

        # Make plots
        _legend_added = False
        for ax, (lev_other, pathis) in zip(axes.flatten(), dict_lev_pa.items()):
            if pathis is not None:
                # plot
                if _legend_added==False:
                    ADD_LEGEND = not len(levels_var_exist)>16
                    _legend_added=True
                else:
                    ADD_LEGEND = False
                pathis.plotwrapper_smoothed_fr_split_by_label("trials", var, ax,
                    event_bounds=event_bounds, legend_levels=levels_var_exist,
                    add_legend=ADD_LEGEND)
            #     self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, xmin=xmin, xmax=xmax)
                ax.set_title(lev_other, fontsize=6, wrap=True)
                # ax.set_ylim([0, 1.2*YMAX])
                ax.axvline(0, color="m")

        return fig, axes

        # else:
        #     fig = None
        #     axes = plot_on_these_axes
        #     assert len(axes)>=len(dict_lev_pa), "not enough axes"
        #
        # # joint he axes
        # if len(axes.flatten())>0:
        #     from pythonlib.tools.plottools import share_axes
        #     share_axes(axes, "y")
        #
        # # Make plots
        # _legend_added = False
        # for ax, (lev_other, pathis) in zip(axes.flatten(), dict_lev_pa.items()):
        #     if pathis is not None:
        #         # plot
        #         if _legend_added==False:
        #             ADD_LEGEND = not len(levels_var_exist)>16
        #             _legend_added=True
        #         else:
        #             ADD_LEGEND = False
        #         pathis.plotwrapper_smoothed_fr_split_by_label("trials", var, ax,
        #             event_bounds=event_bounds, legend_levels=levels_var_exist,
        #             add_legend=ADD_LEGEND)
        #     #     self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, xmin=xmin, xmax=xmax)
        #         ax.set_title(lev_other, fontsize=6, wrap=True)
        #         # ax.set_ylim([0, 1.2*YMAX])
        #         ax.axvline(0, color="m")
        #
        # return fig, axes
        #
        # # else:
        # #     for lev_other, ax in zip(plot_these_levels_of_varsothers, plot_on_these_axes):
        # #         pathis = dict_lev_pa[lev_other]
        # #         pathis.plotwrapper_smoothed_fr_split_by_label("trials", var, ax, event_bounds=event_bounds);
        # #         ax.set_title(lev_other, fontsize=8)
        # #         ax.set_ylim([0, 1.1*YMAX])
        # #         ax.axvline(0, color="m")
        # #     return None, None



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
                    

    def _plotgood_rasters_extract_xmin_xmax(self, xmin=None, xmax=None):
        """ Simple helper: extract window based on durations used for extractio of data
        """
        if xmin is None:
            xmin = self.Params["list_pre_dur"][0]
        if xmax is None:
            xmax = self.Params["list_post_dur"][0]
        return xmin, xmax

    def _plotgood_rasters(self, dfthis, xmin=None, xmax=None, overlay_strokes=True, overlay_events=True,
            ax=None):
        """Plot rasters for all trials of this dataframe
        """
        if self.SNmult is not None:
            assert False, "doesnt work, since plotmod_overlay_trial_events_mult assumes a single session"
        trials = dfthis["trial_neural"].tolist()
        list_spiketimes = dfthis["spike_times"].tolist()
        event_times = dfthis["event_time"].tolist()
        # include_text = False

        sn, _ = self._session_extract_sn_and_trial()
        xmin, xmax = self._plotgood_rasters_extract_xmin_xmax(xmin, xmax)
        if ax is None:
            dur = xmax-xmin
            ncols = 1
            nrows = 1
            fig, axes, kwargs = sn._plot_raster_create_figure_blank(dur, len(dfthis), nrows, ncols)
            # fig, ax = plt.subplots(figsize=(11,4))
            ax = axes[0][0]
        else:
            fig = None
        sn._plot_raster_line_mult(ax, list_spiketimes, ylabel_trials=trials,
            xmin=xmin, xmax=xmax)
        if overlay_events:
            sn.plotmod_overlay_trial_events_mult(ax, trials, event_times, xmin=xmin, xmax=xmax, 
                overlay_strokes=overlay_strokes)

        return fig, ax

    def _plotgood_rasters_split_by_feature_levels(self, ax, dfthis, var, event=None, 
        levels_var=None, xmin=None, xmax=None):
        """
        Plot rasters for this dataset, blocked by levels of a given feature (var). 
        PARAMS:
        - var, variable whos levels iwll plot
        - levels_var, which to plot, in order from bottom to top.
        """
        
        sn, _ = self._session_extract_sn_and_trial()

        if levels_var is None:
            # levels_var = sorted(dfthis[var].unique().tolist())
            levels_var = sort_mixed_type(dfthis[var].unique().tolist())

        # extract
        list_list_st = []
        list_labels = []
        list_list_trials = []
        list_indstroke_flat = []
        list_list_evtimes = []
        list_list_yval = []
        list_list_snidx = []
        ymax_prev = 0
        for lev_var in levels_var:
            dfthisthis = self.dataextract_as_df_good(dfthis=dfthis, event_aligned=event, var=var, var_level=lev_var)
            # assert False, "get event"
            # dfthisthis = dfthis[(dfthis[var]==lev_var) & (dfthis[event]==lev_var)]
            
            # hier params
            list_st = dfthisthis["spike_times"].tolist()
            list_list_st.append(list_st)
            list_labels.append(lev_var)
        
            # Flat params
            trials = dfthisthis["trial_neural"].tolist()
            if "session_idx" in dfthisthis.columns:
                sn_index = dfthisthis["session_idx"].tolist()
            else:
                sn_index = np.nan
            list_list_snidx.append(sn_index)
            # ind_stroke = dfthisthis["ind_taskstroke_orig"].tolist()
            ev_time = dfthisthis["event_time"].tolist()
            list_list_trials.append(trials)
            # list_indstroke_flat.append(ind_stroke)
            list_list_evtimes.append(ev_time)

            yvals = list(range(ymax_prev, ymax_prev+len(trials)))
            list_list_yval.append(yvals)
            ymax_prev+=len(trials)
            
        # 1) Plot all spike times
        overlay_trial_events = self._CONCATTED_SNIPPETS==False
        sn.plot_raster_spiketimes_blocked(ax, list_list_st, list_labels, list_list_trials, 
            list_list_evtimes, xmin = xmin, xmax=xmax, overlay_trial_events=overlay_trial_events)

        # 2) for each session, plot its event times.        
        if self._CONCATTED_SNIPPETS:
            # Overlay each sub sn

            # i flatten
            list_snidx = [x for X in list_list_snidx for x in X]
            list_trials = [x for X in list_list_trials for x in X]
            list_evtimes = [x for X in list_list_evtimes for x in X]
            list_yval = [x for X in list_list_yval for x in X]

            for i, snidx in enumerate(set(list_snidx)):
                sn = self.SNmult.SessionsList[snidx]
                inds = [i for i, x in enumerate(list_snidx) if x==snidx]
                
                # print(list_snidx)
                trialsthis = [list_trials[i] for i in inds]
                yvalsthis = [list_yval[i] for i in inds]
                evtimesthis = [list_evtimes[i] for i in inds]
                ylabel_trials = [f"{snidx}-{t}" for t in trialsthis]

                clear_old_yticks = i==0

                sn.plotmod_overlay_trial_events_mult(ax, trialsthis, evtimesthis, 
                    list_yvals = yvalsthis, ylabel_trials=ylabel_trials,
                    xmin=xmin, xmax=xmax, clear_old_yticks=clear_old_yticks) 




    def plotgood_rasters_split_by_feature_levels_grpbyothervar(self, site, var,
            vars_others, xmin=None, xmax=None):
        """ [GOOD] Makes a grid plot, showing rasters, each grid shows all levels for ver, for a specific
        level of (var_others).
        Uses spiketimes stored in self.DfScalar, instead of
        in Session. 
        """
        # levels_var = SP.DfScalar[var].unique().tolist()
        dftmp, levdat, levels_var = self.dataextract_as_df_conjunction_vars(var, 
                vars_others, site)

        xmin, xmax = self._plotgood_rasters_extract_xmin_xmax(xmin, xmax)

        ntrials = len(dftmp)
        sn, _ = self._session_extract_sn_and_trial()

        if ntrials == 0:
            print("skipping, no data!")
            fig, axes = None, None
        else:
            ncols = 4
            ngrid = len(levdat)
            nrows = int(np.ceil(ngrid/ncols))
            dur = xmax - xmin
            fig, axes, kwargs = sn._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols)

            for ax, (lev_other, dfthis) in zip(axes.flatten(), levdat.items()):
                self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, xmin=xmin, xmax=xmax)
                ax.set_title(lev_other)
                ax.axvline(0, color="m")

        return fig, axes
            

    def plotgood_rasters_split_by_feature_levels_grpbyevent(self, site, var,
            list_events, xmin=None, xmax=None):
        """ [GOOD, in that uses spiketimes stored in self.DfScalar, instead of
        in Session. Makes a grid plot, showing rasters, each grid shows all levels for ver, for a specific
        level of (var_others).
    
        """

        assert False, "in progress, make it like plotgood_rasters_split_by_feature_levels_grpbyothervar"

        # eventdat = ...

        # # plot
        # fig, axes = plt.subplots(3,3, figsize=(20, 20))

        # for ax, (lev_other, dfthis) in zip(axes.flatten(), levdat.items()):
        #     self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, xmin=xmin, xmax=xmax)
        #     ax.set_title(lev_other)
        #     ax.axvline(0, color="m")

    def plotgood_raw_sanity_check(self, ind):
        """ Plot a single row of self.DfScalar, 
        to compare spikes, sm fr, and fr scalar.
        """

        df = self.DfScalar
        fig, ax = plt.subplots()
        st = df.iloc[ind]["spike_times"]
        fr = df.iloc[ind]["fr_sm"].squeeze()
        fr_times = df.iloc[ind]["fr_sm_times"].squeeze()
        fr_scal = df.iloc[ind]["fr_scalar_raw"]
        fr_scal_2 = np.mean(fr)

        pre_dur, post_dur = self._plotgood_rasters_extract_xmin_xmax()

        ax.plot(fr_times, fr);
        ax.plot(st, np.ones(st.shape), 'ok');
        ax.plot(st, fr_scal*np.ones(st.shape), '-r');
        ax.plot(st, fr_scal_2*np.ones(st.shape), '--r');
        # ax.plot(st, fr_scal, '-r');
        ax.axvline(pre_dur)
        ax.axvline(post_dur)

        ax.set_title(f"DfScalar row {ind}")

        print("Mainly check: spikes matches smoothed fr")

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

        sn, _ = self._session_extract_sn_and_trial()

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
                sn.plot_raster_trials_blocked(ax, list_trials_sn, site, list_labels, 
                                           alignto=event_orig,
                                           overlay_trial_events=False, xmin=pre_dur-0.2, 
                                           xmax=post_dur+0.2)

                self.plotmod_overlay_event_boundaries(ax, event, overlay_pre_and_post_boundaries=overlay_pre_and_post_boundaries)


                if j==0:
                    ax.set_title(var)
                if i==0:
                    ax.set_ylabel(event_orig)
        return fig, axes

    def plotgood_rasters(self, site, event=None, ax=None):
        """ Plot a single raster plot for this site and event, not split by any variables.
        """
        dfthis = self.dataextract_as_df_good(chan=site, event_aligned=event)
        fig, ax = self._plotgood_rasters(dfthis, xmin=None, xmax=None, ax=ax)
        return fig, ax

    def plotgood_rasters_smfr_combined(self, site, event=None):
        """
        Basic plot, rasters and smoothed FR for this site, across all datapts,
        without splitting by any var.
        """

        # # PREPARE plot
        # # Extract data, just to see many subplots to make.
        # _, levdat, levels_var = self.dataextract_as_df_conjunction_vars(var, 
        #         vars_others, site, event=event)

        # # if the other_level names are too long, shorten and use that 
        # # if max([len(lov) for lov in levdat.keys()])>MAX_TITLE_LENGTH:
        # #     # replace with indices and redo
        # #     from pythonlib.tools.listtools import map_categoricalvar_to_indices
        # #     map_idx_to_level, map_level_to_idx = map_categoricalvar_to_indices(list(levdat.keys()))
        # # else:
        # #     map_idx_to_level = None
        # #     map_level_to_idx = {lev:lev for lev in levdat.keys()}

        # if len(levdat)>0:
        #     map_level_to_idx = {lev:lev for lev in levdat.keys()}
        #     max_title_len = max([len(str(lov)) for lov in levdat.keys()])
        #     if max_title_len>50:
        #         FONTSIZE = 4
        #     else:
        #         FONTSIZE = 6
        ncols = 1
        dfthis = self.dataextract_as_df_good(site, event)
        ntrials = len(dfthis)
        xmin, xmax = self._plotgood_rasters_extract_xmin_xmax()
        dur = xmax-xmin
        nrows = 2
        sn, _ = self._session_extract_sn_and_trial()
        fig, axesall, kwargs = sn._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols, 
                                                                       reduce_height_for_sm_fr=True)

        # fig, axesall = plt.subplots(2,1)

        # 1) Plot the rasters ont he top row.
        # axes = axesall[0]
        ax = axesall.flatten()[0]
        self.plotgood_rasters(site, event, ax)    
        # # Make sure is readable even if is long name.
        # ax.set_title(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
        # ax.set_ylabel(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
        # assert False
        ax.axvline(0, color="m") 
        

        # 2) Plot the sm fr on the lower row
        ax = axesall.flatten()[1]
        self._plotgood_smoothfr(site, event, ax=ax)
        ax.axvline(0, color="m") 

        # self._plotgood_smoothfr_average_each_level(site, var, vars_others, event=event,
        #                                          plot_these_levels_of_varsothers=levels_of_varsothers,
        #                                          plot_on_these_axes=axes)  
        # make sure x axes is same for raster and sm fr
        for i in range(ncols):
            ax1 = axesall[0][i]
            ax2 = axesall[1][i]
            ax2.set_xlim(ax1.get_xlim())

        # frates, floor at 0
        for i in range(ncols):
            ax2 = axesall[1][i]
            ax2.set_ylim(0)

        return fig, axesall

    def plotgood_rasters_smfr_each_level_combined(self, site, var, 
                vars_others=None, event=None, plotvers=("raster", "smfr"),
                                                  OVERWRITE_n_min=None,
                                                  OVERWRITE_lenient_n=None,
                                                  balance_same_levels_across_ovar=False):
        """ [Good], plot in a single figure both rasters (top row) and sm fr (bottom), aligned.
        Each column is a level of vars_others. 
        PARAMS;
        - var, str, the variable, e.g, epoch"
        --- or list of str, which will be interpreted as conjunction var.
        NOTE:
        - this could be made flexible so that each column is anyting, such as events.

        """
        
        # NOT DONE:
        # if orient=="vert":
        #     ncols = 2
        #     nrows = len(list_events_uniqnames)
        #     sharex = "row"
        #     sharey = "col"
        # elif orient=="horiz":
        #     ncols = len(list_events_uniqnames)
        #     nrows = 2
        #     sharex = "col"
        #     sharey = "row"
        # else:
        #     print(orient)
        #     assert False

        assert event is not None, "or else OVERWRITE_n_min wont work, it will counta cross all event"

        if isinstance(var, (tuple, list)):
            from pythonlib.tools.pandastools import append_col_with_grp_index
            self.DfScalar = append_col_with_grp_index(self.DfScalar, var, "_tmp")
            var = "_tmp"

        if isinstance(vars_others, tuple):
            vars_others = list(vars_others)

        # PREPARE plot
        # Extract data, just to see many subplots to make.
        _, levdat, levels_var = self.dataextract_as_df_conjunction_vars(var,
                vars_others, site, event=event, OVERWRITE_n_min=OVERWRITE_n_min,
                                                  OVERWRITE_lenient_n=OVERWRITE_lenient_n,
                                                    balance_same_levels_across_ovar=balance_same_levels_across_ovar)

        # if the other_level names are too long, shorten and use that
        # if max([len(lov) for lov in levdat.keys()])>MAX_TITLE_LENGTH:
        #     # replace with indices and redo
        #     from pythonlib.tools.listtools import map_categoricalvar_to_indices
        #     map_idx_to_level, map_level_to_idx = map_categoricalvar_to_indices(list(levdat.keys()))
        # else:
        #     map_idx_to_level = None
        #     map_level_to_idx = {lev:lev for lev in levdat.keys()}

        if len(levdat)>0:
            map_level_to_idx = {lev:lev for lev in levdat.keys()}
            max_title_len = max([len(str(lov)) for lov in levdat.keys()])
            if max_title_len>50:
                FONTSIZE = 4
            else:
                FONTSIZE = 6
                
            ntrials = np.mean([len(v) for k, v in levdat.items()])
            ncols = len(levdat)
            xmin, xmax = self._plotgood_rasters_extract_xmin_xmax()
            dur = xmax-xmin
            nrows = 2
            sn, _ = self._session_extract_sn_and_trial()
            fig, axesall, kwargs = sn._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols, 
                                                                           reduce_height_for_sm_fr=True)
 
            # 1) Plot the rasters ont he top row.
            if "raster" in plotvers or "rasters" in plotvers:
                axes = axesall[0]
                for ax, (lev_other, dfthis) in zip(axes.flatten(), levdat.items()):
                    self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, event=event,
                        xmin=xmin, xmax=xmax)
                    # Make sure is readable even if is long name.
                    ax.set_title(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
                    ax.set_ylabel(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
                    ax.axvline(0, color="m")

            # # 2) Plot the sm fr on the lower row
            if "smfr" in plotvers:
                axes = axesall[1]
                levels_of_varsothers = list(levdat.keys())
                self._plotgood_smoothfr_average_each_level(site, var, vars_others, event=event,
                                                         plot_these_levels_of_varsothers=levels_of_varsothers,
                                                         plot_on_these_axes=axes,
                                                           OVERWRITE_n_min=OVERWRITE_n_min, OVERWRITE_lenient_n=OVERWRITE_lenient_n,
                                                                                balance_same_levels_across_ovar=balance_same_levels_across_ovar)

            # make sure x axes is same for raster and sm fr
            for i in range(ncols):
                ax1 = axesall[0][i]
                ax2 = axesall[1][i]
                ax2.set_xlim(ax1.get_xlim())
 
            # frates, floor at 0
            for i in range(ncols):
                ax2 = axesall[1][i]
                ax2.set_ylim(0)
        else:
            if False: # unecessary time.
                fig, axesall = plt.subplots(2,1)
                axesall.flatten()[0].set_title("Not enough data!")
            else:
                print("SKIPPING RASTER - not enough data", site, " - ", var, " - ", vars_others, " - ", event)
                fig, axesall = None, None

        return fig, axesall

    def plotgood_smfr_each_level_subplot_grid_by_vars(self, site, var, var_other_1,
                                                 var_other_2, PLOT_VER="raster", 
                                                      event=None):
        """
        Plot modulation by one variable (var) split into subplots organized in grid,
        where rows (var_other_1) and columns (var_other_2) are variables.
        Two plot versions, switched by PLOT_VER
        PARAMS:
        - PLOT_VER, either "raster" or "smfr"
        """
        from pythonlib.tools.listtools import sort_mixed_type
        from pythonlib.tools.plottools import share_axes

        # PREPARE plot
        # Extract data, just to see many subplots to make.
        vars_others = [var_other_1, var_other_2]
        _, levdat, levels_var = self.dataextract_as_df_conjunction_vars(var,
                vars_others, site, event=event)
        levels_othervar = list(levdat.keys())
        levels_othervar_1 = sort_mixed_type(set([x[0] for x in levels_othervar]))
        levels_othervar_2 = sort_mixed_type(set([x[1] for x in levels_othervar]))

        # Differences, raster vs. smfr
        if PLOT_VER=="smfr":
            force_scale_height = 0.7
            reduce_height_for_sm_fr = True
            sharey = True
        elif PLOT_VER=="raster":
            force_scale_height = None
            reduce_height_for_sm_fr = False
            sharey = False

        if len(levdat)>0:
            map_level_to_idx = {lev:lev for lev in levdat.keys()}
            max_title_len = max([len(str(lov)) for lov in levdat.keys()])
            if max_title_len>50:
                FONTSIZE = 4
            else:
                FONTSIZE = 6

            ntrials = np.mean([len(v) for k, v in levdat.items()])
            xmin, xmax = self._plotgood_rasters_extract_xmin_xmax()
            dur = xmax-xmin
            nrows = len(levels_othervar_1)
            ncols = len(levels_othervar_2)
            sn, _ = self._session_extract_sn_and_trial()
            fig, axesall, kwargs = sn._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols,
                                                                           reduce_height_for_sm_fr=reduce_height_for_sm_fr,
                                                                            sharey=sharey, 
                                                                       force_scale_height=force_scale_height)
            # Go thru each row
            for i, lev1 in enumerate(levels_othervar_1):
                axes = axesall[i]

                vars_others = [var_other_1, var_other_2]
                levels_of_varsothers = [(lev1, lev2) for lev2 in levels_othervar_2]

                if PLOT_VER=="smfr":
                    # Smoothed fr.
                    self._plotgood_smoothfr_average_each_level(site, var, vars_others, event=event,
                                                             plot_these_levels_of_varsothers=levels_of_varsothers,
                                                             plot_on_these_axes=axes, leave_subplot_empty_if_no_data=True)
                    for ax in axes.flatten():
                        ax.set_ylim(0)
                elif PLOT_VER == "raster":
                    # Rasters
                    for ax, lev_other in zip(axes.flatten(), levels_of_varsothers):
                        if lev_other in levdat.keys():
                            dfthis = levdat[lev_other]
                            self._plotgood_rasters_split_by_feature_levels(ax, dfthis, var, event=event,
                                xmin=xmin, xmax=xmax)
                            # Make sure is readable even if is long name.
                            ax.set_title(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
                            ax.set_ylabel(map_level_to_idx[lev_other], fontsize=FONTSIZE, wrap=True)
                            ax.axvline(0, color="m")
                else:
                    print(PLOT_VER)
                    assert False

            # all axes shared
            share_axes(axesall, which="x")
            if sharey:
                share_axes(axesall, which="y")
        else:
            fig, axesall = plt.subplots(2,1)
            axesall.flatten()[0].set_title("Not enough data!")

        # Title tjhem
        axesall.flatten()[0].set_ylabel(var_other_1)
        axesall.flatten()[0].set_xlabel(var_other_2)

        return fig, axesall

    def plot_rasters_smfr_each_level_combined(self, site, var, 
        list_events_uniqnames = None, orient="vert",
        width = 4, height = 3, overlay_pre_and_post_boundaries = True):
        """ plot in a single figure both rasters and sm fr, aligned.
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
        - event, unique string name (prefix num), e.g, 00_go', or '01_doneb
        RETURNS:
        - pre_dur, num
        - post_dur, num
        """

        list_events_uniqnames = self.Params["list_events_uniqnames"]
        list_pre_dur = self.Params["list_pre_dur"]
        list_post_dur = self.Params["list_post_dur"]

        if event not in list_events_uniqnames:
            # Then probably doesnt have numerical prefix. THe following is correct
            tmp = [i for i, x in enumerate(self.Params["_list_events"]) if x==event]
            if len(tmp)!=1:
                print(event)
                print(self.Params)
                print(tmp)
                assert False
            ind = tmp[0]
        else:
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

    ########################### Sites
    def sites_check_span_all_sessions(self):
        """
        [Only for concated SP] Check if each site has data spanning all sessions. If 
        not then throw error
        """
        from pythonlib.tools.pandastools import grouping_get_inner_items
        this = grouping_get_inner_items(self.DfScalar, "chan", "session_idx")
        print(this)
        for site in self.Sites:
            print(site, ":", this[site])
            assert False, "in progress"
            # TODO: check that this[site] is identical to unqiue list of sites

    def animal(self):
        """ Return string, animal"""
        sn = self._session_extract_all()[0]
        return sn.Animal

    def date(self):
        """ Return int, date, YYMMDD"""
        sn = self._session_extract_all()[0]
        return sn.Date

    def bregion_list(self, combine_into_larger_areas=False):
        """ GEt begion list from SEssion()"""
        sn = self._session_extract_all()[0]
        list_bregion = sn.sitegetter_get_brainregion_list_BASE(
            combine_into_larger_areas=combine_into_larger_areas)
        return list_bregion

    def sitegetter_map_region_to_sites(self, bregion,
                                       exclude_bad_areas=False):
        """ Return list of ints (sites) that are in self.Sites,
        and also for this bregion.
        RETURNS:
        - sites, sorted list of int sites.
        """

        # collect sites across all sessions
        list_sn = self._session_extract_all()
        list_sites = []
        for sn in list_sn:
            # list_sites.extend(sn.sitegetter_all([bregion]))
            list_sites.extend(sn.sitegetterKS_map_region_to_sites(bregion, exclude_bad_areas=exclude_bad_areas))

        # Only keep self.Sites which are in that list
        sites = [s for s in self.Sites if s in list_sites]

        return sorted(sites)


    def sites_update_fr_thresh(self, fr_thresh=4, fr_percentile=None, 
        plot_hist=False, do_update=True):
        """ Updates self.Sites to use only sites with mean fr >= fr_thresh
        Moves orig sites to self.SitesOrig, if it doesnt exist
        """
        from pythonlib.tools.pandastools import aggregGeneral

        if fr_thresh:
            assert fr_percentile is None
        if fr_percentile:
            assert fr_thresh is None

        print("starting sites: ", len(self.Sites))
        print("starting sites: ", self.Sites)

        if "fr_scalar_raw" not in self.DfScalar.columns:
            # quickly extract
            frmat = np.concatenate(self.DfScalar["fr_sm"].tolist())
            assert frmat.shape[0] == len(self.DfScalar)
            self.DfScalar["fr_scalar_raw"] = np.mean(frmat,1)

        df_fr = aggregGeneral(self.DfScalar, ["chan"], values=["fr_scalar_raw"])

        # convert percentile to fr threshold.
        if fr_percentile:
            fr_thresh = np.percentile(df_fr["fr_scalar_raw"], fr_percentile)
            print(f"For percentile {fr_percentile}, using this threshold: {fr_thresh}")
            if plot_hist:
                ps = np.linspace(0, 100, 50)
                frs = np.percentile(df_fr["fr_scalar_raw"], ps)
                fig, ax = plt.subplots()
                ax.plot(frs, ps, '-ok')
                ax.set_xlabel('fr')
                ax.set_ylabel('percentile')
                if False:
                    print("percentile -- fr")
                    print(np.c_[ps, frs])

        # (2) Remove low fr sites.
        if plot_hist:
            fig = plt.figure()
            df_fr["fr_scalar_raw"].hist(bins=100)
            plt.axvline(fr_thresh, color="r")

        sites_good = df_fr[df_fr["fr_scalar_raw"]>=fr_thresh]["chan"].tolist()
        sites_bad = df_fr[df_fr["fr_scalar_raw"]<fr_thresh]["chan"].tolist()

        print("sites_good: ", len(sites_good))
        print("sites_bad: ", len(sites_bad))

        if do_update:
            if not hasattr(self, "SitesOrig") or self.SitesOrig is None:
                self.SitesOrig = self.Sites
            self.Sites = [s for s in self.Sites if s in sites_good]

        print('Updates self.Sites')
        print("ending sites: ", len(self.Sites))
        print("ending sites: ", self.Sites)

        return fig

    def datasetbeh_datstrokes_append_column(self, column, DS=None):
        """ Extract values from beh dataset (DatStrokes) for this column,
        where each datapt is a specific (trialcode, stroke_index),
        and append to self.DfScalar, mutating it.
        PARAMS;
        - dataset, if None, then uses the one saved in Snippets.
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        # 1) Extract dataset
        if DS is None:
            DS = self.datasetbeh_extract_dataset(kind="datstrokes")

        assert "stroke_index" in self.DfScalar.columns, "you should not call this function if this which_level is 'trials'"

        # 2) get each (trialcode stroke_index) in self... and then get its value for <column>
        list_tc = self.DfScalar["trialcode"].tolist()
        list_si = self.DfScalar["stroke_index"].tolist()

        # Get the sliced dataframe
        dfslice = DS.dataset_slice_by_trialcode_strokeindex(list_tc, list_si)
        # dfslice = slice_by_row_label(Dataset.Dat, "trialcode", trialcodesthis,
        #     reset_index=True, assert_exactly_one_each=True)

        # No Nones allowed.
        columnthis = "gridloc"
        if sum(dfslice[columnthis].isna())>0:
            print("-----", columnthis)
            print(sum(dfslice[columnthis].isna()))
            print(sum(DS.Dat[columnthis].isna()))
            print(dfslice[dfslice[columnthis].isna()][columnthis])
            print(DS.Dat[DS.Dat[columnthis].isna()][columnthis])
            print(sum(dfslice[columnthis]==None))
            print(sum(DS.Dat[columnthis]==None))
            assert False

        # Assign the values to self
        print("Updating this column of self.DfScalar with Dataset beh:")
        print(column)
        self.DfScalar[column] = dfslice[column].tolist()


    def datasetbeh_datstrokes_append_column_mult(self, columns, DS=None):
        """ [Good] Quick extraction of moultiple columns from Datraset,
        beh dataset (DatStrokes) where each datapt is a specific (trialcode, stroke_index),
        and append to self.DfScalar, mutating it.
        PARAMS;
        - dataset, if None, then uses the one saved in Snippets.
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        if len(columns)==0:
            return None
        else:
            # Must be unique, or else big problem
            columns = list(set(columns))
            columns = [col for col in columns if col not in self.DfScalar.columns]

        # 1) Extract dataset
        if DS is None:
            DS = self.datasetbeh_extract_dataset(kind="datstrokes")

        assert "stroke_index" in self.DfScalar.columns, "you should not call this function if this which_level is 'trials'"

        # 2) get each (trialcode stroke_index) in self... and then get its value for <column>
        list_tc = self.DfScalar["trialcode"].tolist()
        list_si = self.DfScalar["stroke_index"].tolist()

        # Get the sliced dataframe
        dfslice = DS.dataset_slice_by_trialcode_strokeindex(list_tc, list_si)
        # dfslice = slice_by_row_label(Dataset.Dat, "trialcode", trialcodesthis,
        #     reset_index=True, assert_exactly_one_each=True)

        # Assign the values to self
        # self.DfScalar[column] = dfslice[column].tolist()
        print("Appending thse columns... ", columns)
        # self.DfScalar = self.DfScalar.drop([v for v in columns if v in self.DfScalar.columns], axis=1)
        self.DfScalar = self.DfScalar.join(dfslice.loc[:, columns])

    # def datasetbeh_append_column_mult(self, columns, Dataset=None):
    #     """ Extract values from beh dataset, for this column,
    #     and append to self.DfScalar, mutating
    #     PARAMS;
    #     - dataset, if None, then uses the one saved in Snippets.
    #     """
    #     from pythonlib.tools.pandastools import slice_by_row_label
    #
    #     # 1) Extract dataset
    #     if Dataset is None:
    #         Dataset = self.datasetbeh_extract_dataset()
    #
    #     # 2) get each trialcode in self... and then get its value for <column>
    #     trialcodesthis = self.DfScalar["trialcode"].tolist()
    #
    #     # Get the sliced dataframe
    #     dfslice = slice_by_row_label(Dataset.Dat, "trialcode", trialcodesthis, reset_index=True,
    #                                  assert_exactly_one_each=True)
    #
    #     self.DfScalar = self.DfScalar.drop([v for v in columns if v in self.DfScalar.columns], axis=1)
    #     self.DfScalar = self.DfScalar.join(dfslice.loc[:, columns])


    def datasetbeh_append_column_helper(self, list_var, Dataset=None,
                                        DS=None, stop_if_fail=False):
        """ Tries to append each var in list_var, looking thru
        datasetbeh and datasetstrokes. returns success (get all)
        or failure (missed at least one)
        """

        if Dataset is None:
            Dataset = self.datasetbeh_extract_dataset()

        if DS is None:
            DS = self.datasetbeh_extract_dataset(kind="datstrokes")

        success = True
        if True:
            # New method, do in bulk. First figure out which column are using
            # dataset, and whic use datset_strokes. Then run each one time in bulk.
            columns_dataset = []
            columns_dataset_strokes = []
            for var in list_var:
                if var not in self.DfScalar.columns:
                    if DS is not None and var in DS.Dat.columns:
                        columns_dataset_strokes.append(var)
                    elif var in Dataset.Dat.columns:
                        columns_dataset.append(var)
                    else:
                        success = False
                        print("Failed to find this var:", var)
                        print("Len D:", len(Dataset.Dat))
                        if DS is not None:
                            print("Len DS:", len(DS.Dat))
                        if stop_if_fail:
                            return success

            # Do appending of all
            if len(columns_dataset)>0:
                print("... dataset: ", columns_dataset)
                self.datasetbeh_append_column_mult(columns_dataset, Dataset=Dataset)
            if len(columns_dataset_strokes)>0:
                print("... dataset_strokes: ", columns_dataset_strokes)
                self.datasetbeh_datstrokes_append_column_mult(columns_dataset_strokes, DS=DS)

            # # Code to test that this new method is idnetical results to old method (but much faster)
            # columns = ["seqc_0_shape", "seqc_0_loc", "taskconfig_loc", "shape_semantic_labels"]
            # SP.DfScalar.loc[:, columns]
            # df1 = SP.DfScalar.loc[:, columns].copy()
            # for col in columns:
            #     del SP.DfScalar[col]
            #
            # for col in columns:
            #     SP.datasetbeh_append_column(col, Dataset=Dgood)
            # df2 = SP.DfScalar.loc[:, columns].copy()
            # assert np.all(df1==df2)
        else:
            # Old method - takes a long time.
            success = True
            for var in list_var:
                if var not in self.DfScalar.columns:
                    if DS is not None and var in DS.Dat.columns:
                        print("Appending... ", var)
                        self.datasetbeh_datstrokes_append_column(var, DS)
                    elif var in Dataset.Dat.columns:
                        print("Appending... ", var)
                        self.datasetbeh_append_column(var, Dataset)
                    else:
                        success = False
                        print("Failed to find this var:", var)
                        print("Len D:", len(Dataset.Dat))
                        if DS is not None:
                            print("Len DS:", len(DS.Dat))
                        if stop_if_fail:
                            return success

        return success

    def datasetbeh_append_column(self, column, Dataset=None):
        """ Extract values from beh dataset, for this column, 
        and append to self.DfScalar, mutating 
        PARAMS;
        - dataset, if None, then uses the one saved in Snippets.
        """ 
        from pythonlib.tools.pandastools import slice_by_row_label
        
        # 1) Extract dataset
        if Dataset is None:
            Dataset = self.datasetbeh_extract_dataset()            

        # 2) get each trialcode in self... and then get its value for <column>
        trialcodesthis = self.DfScalar["trialcode"].tolist()

        # Get the sliced dataframe
        dfslice = slice_by_row_label(Dataset.Dat, "trialcode", trialcodesthis, reset_index=True,
                                     assert_exactly_one_each=True)

        # Assign the values to self
        print("Updating this column of self.DfScalar with Dataset beh:")
        print(column)
        self.DfScalar[column] = dfslice[column].tolist()

    def datasetbeh_append_column_mult(self, columns, Dataset=None):
        """ Good, faster, method for
        Extract values from beh dataset, for this column,
        and append to self.DfScalar, mutating
        PARAMS;
        - dataset, if None, then uses the one saved in Snippets.
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        if len(columns)==0:
            return None
        else:
            columns = list(set(columns))
            columns = [col for col in columns if col not in self.DfScalar.columns]

        # 1) Extract dataset
        if Dataset is None:
            print(" ** extravcting dataset")
            Dataset = self.datasetbeh_extract_dataset()

        # 2) get each trialcode in self... and then get its value for <column>
        trialcodesthis = self.DfScalar["trialcode"].tolist()

        # Get the sliced dataframe
        print(" ** slice_by_row_label")
        dfslice = slice_by_row_label(Dataset.Dat, "trialcode", trialcodesthis, reset_index=True,
                                     assert_exactly_one_each=True)

        print("Appending thse columns... ", columns)
        # self.DfScalar = self.DfScalar.drop([v for v in columns if v in self.DfScalar.columns], axis=1)
        self.DfScalar = self.DfScalar.join(dfslice.loc[:, columns])

        # # Assign the values to self
        # print("Updating this column of self.DfScalar with Dataset beh:")
        # print(column)
        # if False:
        #     self.DfScalar[column] = dfslice[column].tolist()
        # else:
        #     from pythonlib.tools.pandastools import merge_subset_indices_prioritizing_second
        #     vars = [column]
        #     self.DfScalar = self.DfScalar.drop([v for v in vars if v in self.DfScalar.columns], axis=1)
        #     self.DfScalar = self.DfScalar.join(dfslice.loc[:, [column]])
        #     # merge_subset_indices_prioritizing_second(self.DfScalar, dfslice.loc[:, column])
    def datasetbeh_extract_dataset(self, kind="dataset"):
        """ Extract Dataset object concated across all sessions, for this Snippets.
        Either trial dataset ("dataset") or strokes ("datstrokes").
        If which_level=="trial", then no DS.
        If which_level=="stroke", then D is from sn and DS is that used to construct SP.
        RETURNS:
        - Dall, a single Dataset (a copy)
        """
        if kind=="dataset":
            # each row is trial
            # check that already extracted and is complete
            # def _check_if_got_all_trialcodes(df_to_check):
            #     list_sn = self._session_extract_all()
            #     # tcs = []
            #     tcs_check = df_to_check["trialcode"].tolist()
            #     for sn in list_sn:
            #         tcs_this = sn.Datasetbeh.Dat["trialcode"]
            #         for tc in tcs_this:
            #             if tc not in tcs_check:
            #                 return False
            #     return True

            # if hasattr(self, "Datasetbeh") and _check_if_got_all_trialcodes(self.Datasetbeh.Dat):
            if hasattr(self, "Datasetbeh"):
                # Then good, just reutrn this
                # Around 5ms if do _check_if_got_all_trialcodes, and 200ns if not
                return self.Datasetbeh
            else:
                print("Snippets -- extracting beh dataset for first time! (concatting and tokens preprocess)")
                # Do concat and preprocessing, one time.
                from pythonlib.dataset.analy_dlist import concatDatasets
                list_sn = self._session_extract_all()
                Dall = concatDatasets([sn.Datasetbeh for sn in list_sn])

                if Dall.TokensStrokesBeh is None:
                    for sn in list_sn:
                        print(sn.Datasetbeh.TokensStrokesBeh)
                    print("If print is not None, then this failed in concatDataset")
                    assert False
                    
                # Preprocess dataset
                if False:
                    # No need, since these datasets have each already been preprocessed...
                    Dall._cleanup_preprocess_each_time_load_dataset()

                # Preprocess to get all toekns-related variables, etc.
                Dall.tokens_preprocess_wrapper_good()

                self.Datasetbeh = Dall

                # use this same dataset for all sessions, to that can clear memory
                HACK_KEDAR = True
                if HACK_KEDAR:
                    for sn in list_sn:
                        del sn.Datasetbeh
                        sn.Datasetbeh = self.Datasetbeh

                return self.Datasetbeh
        elif kind in ["datstrokes", "datasetstrokes", "datstroke", "dat_strokes"]:
            # each row is stroke
            # if self.DS is not None:

            # 3/17/24 - New version, insetaed of using cached, regnerate it. This should be same as what is in DfScalar, since
            # I am not saving and loading old SP anymore.

            if self.Params["which_level"] in ["trial", "saccade_fix_on", "fixon", "flex"]:
                # Then no DS is possible.
                return None
            else:
                if self.DS is None:
                    print("GENERATING DS FOR THE FIRST TIME...") 

                    # Then generate for the first time
                    D = self.datasetbeh_extract_dataset()
                    trialcodes = self.DfScalar["trialcode"].unique().tolist()

                    if self.Params["which_level"] in ["stroke", "stroke_off"]:
                        # Just get DS without any pruning.
                        DS = datasetstrokes_extract(D, "all_no_clean")

                    elif self.Params["which_level"] in ["substroke", "substroke_off"]:
                        # Instead of running pipeline, run previously adn sthen load here. This
                        # important -- combines across sessions for computing, which reduces noise a
                        # lot.
                        from pythonlib.dataset.substrokes import load_presaved_using_pipeline
                        DS, _ = load_presaved_using_pipeline(D)

                    else:
                        print(self.Params["which_level"])
                        assert False

                    # Filter the trials
                    DS.dataset_prune_by_trialcodes(trialcodes)

                    # Sanity check that all rows in SP match the row in DS (tc, si) --> event_time
                    dftmp = self.DfScalar.groupby(["trialcode", "stroke_index"])["event_time"].mean().reset_index()

                    if self.Params["which_level"] in ["stroke", "substroke"]:
                        col = "time_onset"
                    elif self.Params["which_level"] in ["stroke_off", "substroke_off"]:
                        col = "time_offset"
                    else:
                        assert False
                    onsets_ds = DS.dataset_slice_by_trialcode_strokeindex(dftmp.loc[:, "trialcode"].tolist(), dftmp.loc[:, "stroke_index"].tolist())[col]

                    times1 = np.array(dftmp["event_time"])
                    times2 = np.array(onsets_ds)

                    from pythonlib.tools.nptools import isnear
                    if not isnear(times1, times2):
                        print(np.argwhere((times2 - times1)>0.01))
                        assert False, "times in SP and DS do not match! explanations: (i) you loaded old SP. do not do that. (ii) weirndess due to substrokes?"

                    self.DS = DS
                    return self.DS
                else:
                    return self.DS

            # else:
            #     from pythonlib.dataset.dataset_strokes import concat_dataset_strokes
            #     list_ds = [ds for ds in self.DSmult]
            #     DS = concat_dataset_strokes(list_ds)
            #     return DS
        else:
            assert False

    def datasetbeh_preprocess_clean_by_expt(self, ANALY_VER, vars_extract_append,
                                            substrokes_plot_preprocess=True,
                                            HACK_RENAME_SHAPES=False):
        """ [GOOD]
        Prune snippets before running any analyses.
        NOTE: removes all trialcodes that miss even just one stroke (as
        tested in DatStrokes)
        :return:
        - Modifies self.DfScalar,, removing rows based on pruned dataset, and appending
        columns as needed.
        - D, pruned Dataset. Note: this is NOT saved in SP.
        - list_features_extraction, list of str, features that wqere extracted successfulty.
        """
        from neuralmonkey.metadat.analy.anova_params import dataset_apply_params

        print(" ++++ USING THIS ANALY_VER:", ANALY_VER)
        print(vars_extract_append, substrokes_plot_preprocess, HACK_RENAME_SHAPES)
        # get back all the outliers, since they just a single removed outlier (chan x trial) will throw out the entire trial.
        print("Appending outliers...")
        self.datamod_append_outliers()

        print("Appending index datrapts...")
        self.datamod_append_unique_indexdatapt()

        if ANALY_VER!="MINIMAL":
            self.datamod_append_unique_indexdatapt()

        ########################## PREP/CLEAN DATASET
        D = self.datasetbeh_extract_dataset()
        animal = self.animal()
        date = self.date()

        # NOTE: This is only used if doing substrokes.
        # Otherwise DS is generated from D.
        DS = self.datasetbeh_extract_dataset("datstrokes")

        # Preprocess, clean, and prune D
        print("Running dataset_apply_params...")
        D, DS_for_pruning_D, params = dataset_apply_params(D, DS, ANALY_VER, animal, date, save_substroke_preprocess_figures=substrokes_plot_preprocess) # prune it
        del DS

        ############################# PRUNE DFSCALAR using Dataset and DatasetStrokes
        # Prune DfScalar to only have trialcodes that remain after pruning.
        TRIALCODES_KEEP = D.Dat["trialcode"].tolist()
        print("Starting len dfscalar: ", len(self.DfScalar))
        self.DfScalar = self.DfScalar[self.DfScalar["trialcode"].isin(TRIALCODES_KEEP)].reset_index(drop=True)
        print("Ending len dfscalar: ", len(self.DfScalar))

        # Optioanlly, for which_level = "stroke", prune rows based on (tc, stroke index).
        if self.Params["which_level"] in ["stroke", "stroke_off", "substroke", "substroke_off"] and DS_for_pruning_D is not None:
            tcs = DS_for_pruning_D.Dat["trialcode"].tolist()
            sis = DS_for_pruning_D.Dat["stroke_index"].tolist()
            print(" --- Pruning SP.DfScalar to match DS...", "start len: ", len(self.DfScalar))

            # assert_exactly_one_each = False, since a trial can have multiple rows in SP.DfScalar.
            from pythonlib.tools.pandastools import append_col_with_grp_index
            DS_for_pruning_D.Dat = append_col_with_grp_index(DS_for_pruning_D.Dat, ["trialcode", "stroke_index"], new_col_name="trialcode_strokeidx",
                                     use_strings=False)
            self.DfScalar = append_col_with_grp_index(self.DfScalar, ["trialcode", "stroke_index"], new_col_name="trialcode_strokeidx",
                                     use_strings=False)
            self.DfScalar = DS_for_pruning_D.dataset_slice_by_trialcode_strokeindex(tcs, sis, df=self.DfScalar,
                                                                          assert_exactly_one_each=False)
            print("... End len: ", len(self.DfScalar))

        ###################### EXTRACT FINAL DS THAT PLACE INTO SP, which will be used to extract featues. THis must run
        # (New, since I am now extracting SP anew, instead of loading previously saved).
        # after all preprocessing of D is done.
        self.Datasetbeh = D # Replace
        # Generate DS newly, from D
        self.DS = None
        DS = self.datasetbeh_extract_dataset("datstrokes")

        # Delete other places where copies of D might exist
        if False: # not good, since you might need D in sn later
            for sn in self.SNmult.SessionsList:
                del sn.Datasetbeh

        ################ CHUNKS, STROKES (e.g., singlerule, AnBm)
        if params["datasetstrokes_extract_chunks_variables"] and self.Params["which_level"] == "stroke":
            # First extract within Dataset, then to DS.
            # (note: DS.Dataset and D are identical objects).

            if False:
                # First, place preprocessed D (from above) into DS.
                DS.dataset_replace_dataset(D)
                # Then prune DS to match D.
                DS.dataset_prune_self_to_match_dataset()

            # Third, extract variables to strokes
            DS.context_chunks_assign_columns()

        ####### PREPROCESSING FOR DS THAT SHOULD ALWAYS RUN:
        if DS is not None:
            # append Tkbeh_stktask
            DS.tokens_append(ver="beh_using_task_data")

        ############### RETURN, IF MINIMAL (kgg, fixations).
        if ANALY_VER == "MINIMAL":
            return D, []

        ################ SUBSTROKES
        if params["substrokes_features_do_extraction"]:
            # Moved out here since now DS is recreated out here. BUT this is not working, since 
            # substrokes do the thing of saving and loading DS, which is not cufrrent method.
            from pythonlib.dataset.substrokes import features_motor_extract_and_bin
            save_substroke_preprocess_figures = substrokes_plot_preprocess
            assert params["datasetstrokes_extract_to_prune_stroke_and_get_features"] is None, "they would overwrite each other"

            # Save in substrokes preprocess folder.
            if save_substroke_preprocess_figures: # Takes too long
                SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("substrokes_preprocess")
                plot_save_dir = f"{SAVEDIR}/plots_during_anova_params"
                os.makedirs(plot_save_dir, exist_ok=True)
            else:
                plot_save_dir = None

            # Extract motor variables (DS)
            features_motor_extract_and_bin(DS, plot_save_dir=plot_save_dir)

        ##################################### EXTRACT FEATURES INTO DFSCALAR
        list_features_extraction_base = ["trialcode", "aborted", "event_time", "task_kind", "gridsize",
                                          "FEAT_num_strokes_task", "FEAT_num_strokes_beh",
                                          "character", "probe", "supervision_stage_concise", "superv_COLOR_METHOD",
                                         "INSTRUCTION_COLOR", "epoch_orig", "epoch", "taskgroup", "epochset",
                                          "origin", "donepos"]

        # These are in both stroek and substroke (the latter is loaded from a saved pickled DS, so it only has older stuff)
        list_features_extraction_stroke_substroke = ["stroke_index", "stroke_index_fromlast", "stroke_index_fromlast_tskstks",
                                    "stroke_index_semantic", "stroke_index_semantic_tskstks",
                                    "shape_oriented", "shape", "gridloc", "gridloc_x", "gridloc_y",
                                    "CTXT_loc_next", "CTXT_shape_next",
                                    "CTXT_loc_prev", "CTXT_shape_prev",
                                    "gap_from_prev_angle_binned", "gap_to_next_angle_binned",
                                    "gap_from_prev_angle", "gap_to_next_angle",
                                    "Stroke"]

        list_features_extraction_stroke = [
                                    "TokTask", "stroke_index_is_first", "stroke_index_is_last_tskstks",
                                    "loc_on_clust", "CTXT_loconclust_prev", "CTXT_loconclust_next",
                                    "loc_off_clust", "CTXT_locoffclust_prev", "CTXT_locoffclust_next",
                                    "shape_semantic", "shape_semantic_cat"]
                                    # "shape_is_novel"]
                                    # "distcum", "displacement", "circularity"]
                                    # "distcum", "displacement", "circularity"]

        list_features_extraction_substroke = ["circ_signed", "velocity", "distcum", "angle",
                                              "distcum_binned", "angle_binned", "circ_signed_binned",
                                              "velocity_binned", "di_an_ci_ve_bin",
                                              "ss_this_ctxt", "CTXT_prev_next", "CTXT_prev_this_next"]
        # supervision_stage_new, trial_neural, "char_seq"

        list_features_extraction_trial = ["shape_is_novel_all", "shape_semantic_labels", "shape_is_novel_list",
                                          "taskconfig_shp", "taskconfig_shploc", "taskconfig_loc",
                                          "Tkbeh_stkbeh", "Tkbeh_stktask", "Tktask",
                                          "taskconfig_shp_SHSEM", "taskconfig_shploc_SHSEM"                                          ]
        n_strok_max = 6
        for i in range(n_strok_max):
            # for suff in ["shape", "loc", "loc_local"]:
            # for suff in ["shape", "loc"]:
            for suff in ["shape", "loc", "shapesem", "locon", "locx", "locy",
                         "center_binned", "locon_binned", "shapesemcat", "shapesemgrp", "angle", "angle_binned",
                         "loc_on_clust"]:
                # locon_bin_in_loc
                list_features_extraction_trial.append(f"seqc_{i}_{suff}")
        list_features_extraction_trial.append("seqc_nstrokes_beh")
        list_features_extraction_trial.append("seqc_nstrokes_task")

        # SANITY CEHCK that there are no identical names ... this leads to ambiguity as to which dataset to get this var from
        assert not any([f in list_features_extraction_stroke_substroke for f in list_features_extraction_trial])
        assert not any([f in list_features_extraction_stroke for f in list_features_extraction_trial])
        assert not any([f in list_features_extraction_substroke for f in list_features_extraction_trial])

        # Features that should always extract (Strokes dat)
        if self.Params["which_level"] in ["substroke", "substroke_off"]:
            print("Using substroke...")
            list_features_extraction = list_features_extraction_base + list_features_extraction_stroke_substroke + list_features_extraction_substroke
            DS_for_feature_extraction = DS
            assert DS is not None
        elif self.Params["which_level"] in ["stroke", "stroke_off"]:
            print("Using stroke...")
            list_features_extraction = list_features_extraction_base + list_features_extraction_stroke_substroke + list_features_extraction_stroke
            DS_for_feature_extraction = DS
            assert DS is not None
        elif self.Params["which_level"]=="trial":
            print("Using trial...")
            list_features_extraction = list_features_extraction_base + list_features_extraction_trial
            DS_for_feature_extraction = None
        elif self.Params["which_level"] in ["saccade_fix_on", "fixon", "flex"]:
            list_features_extraction = list_features_extraction_base
            DS_for_feature_extraction = None
        else:
            print(self.Params["which_level"])
            assert False

        # NOTE: this is older stuff!! currently in Dataset the shapes are already updated in tokens.. so dont need
        # to do this.
        # Extract labels related to char stroke shapes
        if params["charclust_dataset_extract_shapes"] and self.Params["which_level"]=="trial":
            # Then this means it was extracted (trial-level)
            list_features_extraction = list_features_extraction + ["charclust_shape_seq", "charclust_shape_seq_scores"]

        if params["datasetstrokes_extract_to_prune_stroke_and_get_features"] in ["chars_load_clusters", "clean_chars_load_clusters"]  and not self.Params["which_level"]=="trial":
            # Then dataset_strokes lloaded char labels
            list_features_extraction = list_features_extraction + ["shape_label", "clust_sim_max", "clust_sim_max_colname"]

        if params["datasetstrokes_extract_chunks_variables"] and self.Params["which_level"] == "stroke":
            # Then dataset_strokes lloaded chunk variables, e.g,
            list_features_extraction = (list_features_extraction + ["chunk_rank", "chunk_within_rank", "chunk_within_rank_semantic",
                                                                    "chunk_within_rank_semantic_v2", "chunk_within_rank_semantic_v3",
                                                        "chunk_within_rank_fromlast", "chunk_n_in_chunk", "epoch_rand",
                                                        "chunk_diff_from_prev"] + ["taskcat_by_rule", "behseq_shapes"] +
                                        ["syntax_concrete", "syntax_role"] + ["epoch_orig_rand_seq", "epoch_is_AnBmCk", "epoch_is_DIR",
                                                                              "superv_is_seq_sup", "INSTRUCTION_COLOR"]
                                        + ["epochset_diff_motor", "epochset_shape", "epochset_dir"]
                                        )

            # Add concrete variations within each taskcat_by_rule
            LIST_VAR_BEHORDER=["behseq_shapes", "behseq_locs", "behseq_locs_x", "behseq_locs_diff", "behseq_locs_diff_x"]
            list_features_extraction.extend([f"{var_behorder}_clust" for var_behorder in LIST_VAR_BEHORDER])

            # Add bin indicating wheterh chunk gap is short or long duration
            tmp = ["chunkgap_(0, 1)_durbin", "chunkgap_(1, 2)_durbin"]
            for t in tmp:
                if t in D.Dat.columns:
                    list_features_extraction.append(t)

        # For the rest, try to get automatically.
        list_features_extraction = vars_extract_append + list_features_extraction
        list_features_extraction = list(set(list_features_extraction))

        print("Attempting to extract these features into Snippets:")
        print(list_features_extraction)

        # HACKY - remove all None from gridloc.
        print("cleanpreprocessifreloaded")
        if DS_for_feature_extraction is not None:
            DS_for_feature_extraction.clean_preprocess_if_reloaded()

        # Perform extraction
        print("assert datasetbehappcolhelp")
        HACK=False # was crashing because of memory issues
        if not HACK:
            assert self.datasetbeh_append_column_helper(list_features_extraction, D, DS=DS_for_feature_extraction, stop_if_fail=True)==True # Extract all the vars here

            print("listfeaturesextraction")
            for f in list_features_extraction:
                if f not in self.DfScalar.columns:
                    print(f)
                    print(self.DfScalar.columns)
                    assert False
            # Sanity check that no Nones... Had this issue at one point.
            # if "gridloc" in self.DfScalar.columns:
            #     if sum(self.DfScalar["gridloc"].isna())>0:
            #         from pythonlib.tools.pandastools import replace_values_with_this
            #         replace_values_with_this(self.DfScalar, "gridloc", None, ("IGN", "IGN"))
            #         # print("DS:", DS_for_feature_extraction)
            #         # print(DS_for_feature_extraction.Dat["gridloc"].value_counts())
            #         # print(sum(DS_for_feature_extraction.Dat["gridloc"].isna()))
            #         # print(DS_for_feature_extraction.Dat[DS_for_feature_extraction.Dat["gridloc"].isna()])
            #         # assert False, "Fix this above... see what is done in DatStrokes.clean_preprocess_if_reloaded."
            print("replacevaluesgridseqc")
            from pythonlib.tools.pandastools import replace_values_with_this
            if "gridloc" in self.DfScalar.columns:
                replace_values_with_this(self.DfScalar, "gridloc", None, ("IGN", "IGN"))
            if "seqc_0_loc" in self.DfScalar.columns:
                replace_values_with_this(self.DfScalar, "seqc_0_loc", None, ("IGN", "IGN"))

            # # Check that all extracted...
            # for var in list_features_extraction:
            #     if var not in self.DfScalar.columns:
            #         print(var)
            #         print(D.Dat.columns)
            #         print(self.DfScalar.columns)
            #         assert False

            if False:
                print(" ------------ ")
                for col in self.DfScalar.columns:
                    print(col, " -- ", type(self.DfScalar[col].values[0]), " -- ", self.DfScalar[col].values[0])

                print(" *********** ", type(D.Dat["seqc_0_shape"]), type(D.Dat["seqc_0_shape"].values[0]), " -- ", D.Dat["seqc_0_shape"].values[0])
                Dthis = self.datasetbeh_extract_dataset()
                print(" *********** ", type(Dthis.Dat["seqc_0_shape"]), type(Dthis.Dat["seqc_0_shape"].values[0]), " -- ", Dthis.Dat["seqc_0_shape"].values[0])

            ################### OTHER FINAL AD-HOC TOUCHES TO SP DATA.
            # Make a general-purpose "shape" columns applies across trial and stroke data.
            # try:
            print("settingdfscalarstuff")
            self.DfScalar["size_this_event"] = self.DfScalar["gridsize"]
            if self.Params["which_level"] == "trial":
                # trial is isually just up to first stroke...
                # print("---------------")
                # print(self.DfScalar["seqc_0_shape"].unique())
                # print("shape_this_event" in list(self.DfScalar.columns))
                self.DfScalar["shape_this_event"] = self.DfScalar["seqc_0_shape"]
                self.DfScalar["loc_this_event"] = self.DfScalar["seqc_0_loc"]
                list_features_extraction.append("shape_this_event")
                list_features_extraction.append("loc_this_event")
            elif self.Params["which_level"] in ["stroke", "stroke_off"]:
                self.DfScalar["shape_this_event"] = self.DfScalar["shape_oriented"]
                self.DfScalar["loc_this_event"] = self.DfScalar["gridloc"]
                list_features_extraction.append("shape_this_event")
                list_features_extraction.append("loc_this_event")
            elif self.Params["which_level"] in ["substroke", "substroke_off"]:
                self.DfScalar["shape_this_event"] = self.DfScalar["shape"]
                self.DfScalar["loc_this_event"] = self.DfScalar["gridloc"] # HACKY - should actualyl be location of substroke, but not ready
                list_features_extraction.append("shape_this_event")
                list_features_extraction.append("loc_this_event")
            elif self.Params["which_level"] in ["saccade_fix_on", "fixon", "flex"]:
                pass
            else:
                print(self.Params)
                print(self.DfScalar.columns)
                assert False
            list_features_extraction.append("size_this_event")
            # except Exception as err:
            #     print(self.Params)
            #     print(list(self.DfScalar.columns))
            #     raise err

            if HACK_RENAME_SHAPES:
                ############# HACK - rename shapes (lumping)
                #### Rename any variable values? Hacky
                # Lump together (done by hand)
                map_shapelump_to_shapes = {}
                map_shape_to_shapelump = {}
                for grp in [
                    ["V-2-1-0", "arcdeep-4-1-0", "usquare-1-1-0", "L|arcdeep-4-1-0"],
                    ["V-2-3-0", "arcdeep-4-3-0", "usquare-1-3-0"],
                    ["V-2-2-0", "arcdeep-4-2-0", "usquare-1-2-0"],
                    ["V-2-4-0", "arcdeep-4-4-0", "usquare-1-4-0", "L|V-2-4-0"],
                    ["squiggle3-3-1-0", "zigzagSq-1-2-0"],
                    ["squiggle3-3-1-1", "zigzagSq-1-2-1"],
                    ["squiggle3-3-2-0", "zigzagSq-1-1-0"],
                    ["squiggle3-3-2-1", "zigzagSq-1-1-1"],
                ]:

                    # name it after the first
                    name = f"L|{grp[0]}"
                    assert name not in map_shapelump_to_shapes
                    map_shapelump_to_shapes[name] = grp

                    for g in grp:
                        assert g not in map_shape_to_shapelump
                        map_shape_to_shapelump[g] = name

                # Replace shape values for all columns that have "shape" in them.
                shape_keys = [k for k in list_features_extraction if "shape" in k]
                for sk in shape_keys:
                    if isinstance(self.DfScalar.iloc[0][sk], str):
                        if len(self.DfScalar[sk].unique())>3:
                            print(" -- Lumping shapes (renaming) in self.DfScalar, for: ", sk)
                            def F(x):
                                sh = x[sk]
                                if sh in map_shape_to_shapelump.keys():
                                    return map_shape_to_shapelump[sh]
                                else:
                                    return sh
                            self.DfScalar[sk] = self.DfScalar.apply(F, axis=1)

        # print("adding saccade-fixation columns...")
        # # add columns from 240307_sequence_rasters.ipynb
        # self._addSaccadeFixationColumns()

        return D, list_features_extraction

    # adds additional columns for SP.DfScalar here (including code to loop, and append_column vars)
    # NOTE: assumes multiple session SP, e.g. from
    # note: if change to add new columns, must change list_features_extraction in dfallpa_extraction_load_wrapper_from_MS
    def _add_clusterfix_saccfix_columns(self, filter_outliers=False, filter_only_first_shapefix=False):
        import math
        # get the start, end times for the window spanned by start_event, end_event
        def getTimeWindowOfEvents(sn, trial, start_event, end_event):
            # keep just times between [start_event, end_event]
            dict_event_times = sn.events_get_time_sorted(trial, list_events=(start_event, end_event))[0]
            start_time = dict_event_times[start_event][0]
            end_time = dict_event_times[end_event][0]

            return start_time, end_time

        # array of tokens, each one is a task stroke with info such as shapename etc.
        def getAllTaskStrokeTokens(sn, trial):
            dataset_index_from_neural = sn.datasetbeh_trial_to_datidx(trial)
            return sn.Datasetbeh.taskclass_tokens_extract_wrapper(dataset_index_from_neural, "task", plot=False)

        ############
        ## SHAPES ##
        ############
        # def getShapesInOrder(sn, trial):
        #     ts = getAllTaskStrokeTokens(sn, trial)

        #     shape_names = []
        #     for i, t in enumerate(ts):
        #         shape_name = t['shape'] + '-' + str(i) # adds unique tag onto it, so same shape is named differently
        #         shape_names.append(t['shape'])

        #     return shape_names

        # def getShapeCentroidsInOrder(sn, trial):
        #     shape_coords = sn.strokes_task_extract(trial)
        #     shape_names = getShapesInOrder(sn, trial)
        #     shape_centroids = {}
        #     for i in range(len(shape_names)):
        #         name = shape_names[i]
        #         centroid = [np.mean(shape_coords[i][:,0]), np.mean(shape_coords[i][:,1])]
        #         #print("name", name)
        #         #print("centroid", centroid)
        #         shape_centroids[name] = centroid

        #     return shape_centroids # returns dict {name: [x,y]}

        # def getClosestShapeToCentroid(sn, trial, centroid, outlier_threshold=600):
        #     x = centroid[0]
        #     y = centroid[1]
        #     shapeDict = getShapeCentroidsInOrder(sn, trial)
        #     distances = []
        #     names = []

        #     for name in shapeDict:
        #         names.append(name)
        #         distances.append(math.dist(shapeDict[name], [x,y]))

        #     shape_ind = np.argmin(distances)
        #     if distances[shape_ind] >= outlier_threshold:
        #         return 'OFFSCREEN'
        #     return names[shape_ind]

        def getClosestShapeAndLocToCentroid(sn, trial, centroid, event_time):
            import pandas as pd
            from pythonlib.tools.distfunctools import closest_pt_twotrajs

            # Params - criteria for assigining a shape to fiaation, based on distance
            MIN_DIST_TASK_TO_FIX = 70 # radius (from closest point along stroke stroke task image), fixation must be within this to assign to this shape (L2)
            MIN_TIME_REL_STIM_ONSET = 0.15 # (saccades take ~0.05-0.1 sec. Account for 0.1 reaction time)

            t_stim_onset = sn.events_get_time_helper("stim_onset", trial, assert_one=True)[0]
            t_go = sn.events_get_time_helper("go", trial, assert_one=True)[0]

            # if outside prep window, then set to ignore value
            if (event_time < t_stim_onset) or (event_time > t_go):
                return "OUTSIDE_PREP_WINDOW", "OUTSIDE_PREP_WINDOW"

            ind = sn.datasetbeh_trial_to_datidx(trial)
            Tk = sn.Datasetbeh.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)

            # For this fixation/centroid, get its distance to task shapes.

            # get distance to each token
            dist_to_each_token = []
            for tk in Tk.Tokens:
                pts = tk["Prim"].Stroke()[:,:2]
                mindist, _, _ = closest_pt_twotrajs(pts, centroid[None, :])
                dist_to_each_token.append(mindist)

            # find the closest token
            idx_min = np.argmin(dist_to_each_token)
            val_min = np.min(dist_to_each_token)

            # For each fixatoin, get its time relative to stim onset.
            time_relative_stim_onset = event_time - t_stim_onset

            if (time_relative_stim_onset>MIN_TIME_REL_STIM_ONSET) & (val_min<MIN_DIST_TASK_TO_FIX):
                # then assign the shape it was lookinga t
                tk = Tk.Tokens[idx_min]
                # - pull out useful things
                shape = tk["shape"]
                gridloc = tk["gridloc"]
            else:
                shape = "FAR_FROM_ALL_SHAPES"
                gridloc = "FAR_FROM_ALL_LOCS"       

            return shape, str(gridloc)

        # def getClosestShapeLocToCentroid(sn, trial, centroid, outlier_threshold=600):
        #     return str(getClosestShapeToCentroid(sn, trial, centroid, outlier_threshold) + getClosestLocToCentroid(sn, trial, centroid, outlier_threshold))

        ###############
        ## LOCATIONS ##
        ###############
        # def getLocationsAndCentroidsInOrder(sn, trial):
        #     ts = getAllTaskStrokeTokens(sn, trial)

        #     loc_names = []
        #     loc_coords = []
        #     for t in ts:
        #         loc_names.append(str(t['gridloc']))
        #         loc_coords.append(t['center'])

        #     return loc_names, loc_coords


        # def getClosestLocToCentroid(sn, trial, centroid, outlier_threshold=600):
        #     x = centroid[0]
        #     y = centroid[1]
        #     locs = getLocationsAndCentroidsInOrder(sn, trial)
        #     locNames = locs[0]
        #     locCentroids = locs[1]

        #     distances = []
        #     names = []

        #     for i, name in enumerate(locNames):
        #         names.append(name)
        #         distances.append(math.dist(locCentroids[i], [x,y]))

        #     loc_ind = np.argmin(distances)
        #     if distances[loc_ind] >= outlier_threshold:
        #         return 'OFFSCREEN'
        #     return str(names[loc_ind])
        # -----------------------------------------------------------


        ## add drawing sequence information directly to SP
        D = self.datasetbeh_extract_dataset()
        D.seqcontext_preprocess()
        list_features_extraction_seq = ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", "seqc_2_shape", "seqc_2_loc", "seqc_3_shape", "seqc_3_loc", "seqc_0_loc_on_clust", "seqc_1_loc_on_clust", "seqc_2_loc_on_clust", "seqc_3_loc_on_clust"]
        self.datasetbeh_append_column_helper(list_features_extraction_seq, D, stop_if_fail=True)
        print("finished appending seqc0 columns..")
        sessions = self.DfScalar['session_idx'].unique()
        dummy_df_all = pd.DataFrame()
        print("looping through sesssions")
        for sesh in sessions:
            sn = self.SNmult.SessionsList[sesh]

            dummy_df = self.DfScalar[self.DfScalar['session_idx']==sesh].copy()
            #print("dummy_df", dummy_df)
            neuraltrials = dummy_df['trial_neural'].unique()
            print("looping through neuraltrials)")
            for nt in neuraltrials:
                # get time window
                start_time, end_time = getTimeWindowOfEvents(sn, nt, "stim_onset", "go")
                middle_time = np.mean([start_time, end_time])

                event_inds_within_trial = dummy_df[dummy_df['trial_neural']==nt]['event_idx_within_trial'].unique()
                print("looping through event inds")
                for eind in event_inds_within_trial:
                    e_df_inds = dummy_df.index[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)]
                    e_df_temp = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)].copy()
                    e_prev_df_temp = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==(eind-1))].copy()

                    ## add between-stimonset and go, early or late planning
                    e_time = e_df_temp.iloc[0]['event_time']
                    print("between stimonsetgo")
                    if (e_time >= start_time) and (e_time <= end_time):
                        dummy_df.loc[e_df_inds, 'between-stimonset-and-go'] = True
                        if (e_time <= middle_time):
                            dummy_df.loc[e_df_inds, 'early-or-late-planning-period'] = 'early'
                        else:
                            dummy_df.loc[e_df_inds, 'early-or-late-planning-period'] = 'late'
                    else:
                        dummy_df.loc[e_df_inds, 'between-stimonset-and-go'] = False
                        dummy_df.loc[e_df_inds, 'early-or-late-planning-period'] = 'IGNORE'

                    ## add shape-fixation and loc-fixation

                    # first, get centroid of this fixation event
                    e_cntrd = e_df_temp.iloc[0]['fixation-centroid']

                    # now, get closest shape to this centroid and add to dummy_df
                    shapefix, locfix = getClosestShapeAndLocToCentroid(sn, nt, e_cntrd, e_time)
                    #shapefix = getClosestShapeToCentroid(sn, nt, e_cntrd)
                    print("shapefixation: ", shapefix)
                    dummy_df.loc[e_df_inds, 'shape-fixation'] = shapefix
                    print("finished shapefix")

                    ## add loc-fixation
                    #locfix = getClosestLocToCentroid(sn, nt, e_cntrd)
                    print("locfixation: ", locfix)
                    dummy_df.loc[e_df_inds, 'loc-fixation'] = locfix
                    print("finished locfix")
                    ## add first-fixation-on-shape 
                    # reset values
                    #e_df_inds = dummy_df.index[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)]
                    e_df_temp = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)].copy()
                    e_prev_df_temp = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==(eind-1))].copy()
                    print("dftemplen", len(e_df_temp))
                    print("prevdftemplen", len(e_prev_df_temp))
                    print("doing firstifxonshape, eind: ", eind)

                    # sometimes, first fixation is within planning window..
                    if shapefix=='FAR_FROM_ALL_SHAPES':
                        dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = 'IGNORE'
                    elif e_df_temp.iloc[0]['between-stimonset-and-go']==True and eind==0:
                        print("0")
                        dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = True
                    elif e_df_temp.iloc[0]['between-stimonset-and-go']==True and e_prev_df_temp.iloc[0]['between-stimonset-and-go'] == False:
                        print("1")
                        dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = True
                    elif e_df_temp.iloc[0]['between-stimonset-and-go']==False:
                        print("2")
                        dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = 'IGNORE'
                    else:
                        print("3")
                        # check if shape for previous eind is the same
                        shape_prev = e_prev_df_temp.iloc[0]['shape-fixation']

                        if shape_prev == shapefix:
                            dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = False
                        else:
                            dummy_df.loc[e_df_inds, 'first-fixation-on-shape'] = True

                        # add prev shape fixation
                        dummy_df.loc[e_df_inds, 'prev-shape-fixation'] = shape_prev
                        # for good measure, add prev loc fixation
                        dummy_df.loc[e_df_inds, 'prev-loc-fixation'] = e_prev_df_temp.iloc[0]['loc-fixation']

                    ## is fixated on first shape drawn?
                    print("doing isfixatedonseqc0shape")
                    dummy_df.loc[e_df_inds, 'is-fixated-on-seqc0shape'] = (e_df_temp.iloc[0]['shape-fixation'] == e_df_temp.iloc[0]['seqc_0_shape'])
                # start counter, for first-fixation-on-shape
                counter = 1
                macrosacc_seq_shape = []
                macrosacc_seq_loc = []
                print("doing macrosaccs")
                for eind in event_inds_within_trial:
                    e_df_inds = dummy_df.index[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)]
                    e_df_temp = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind)]

                    ## add shape-macrosaccade-index
                    if e_df_temp.iloc[0]['between-stimonset-and-go'] == True and e_df_temp.iloc[0]['first-fixation-on-shape'] == True:
                        dummy_df.loc[e_df_inds, 'shape-macrosaccade-index'] = counter
                        counter += 1
                    else:
                        dummy_df.loc[e_df_inds, 'shape-macrosaccade-index'] = -1 # when filtering out first-fixation-on-shape==False, will be remove

                    ## add saccade-dir-angle and saccade-dir-bin
                    if eind==event_inds_within_trial[0]: # first fixation has no direction..
                        dummy_df.loc[e_df_inds, 'saccade-dir-angle'] = 0 # better placeholder value?
                        continue

                    # now, get fixation centroid
                    e_fix_centroid = e_df_temp.iloc[0]['fixation-centroid']
                    x = e_fix_centroid[0]
                    y = e_fix_centroid[1]

                    # and same for previous EVENT IDX
                    e_df_inds_prev = dummy_df.index[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind-1)]
                    e_df_temp_prev = dummy_df[(dummy_df['trial_neural']==nt) & (dummy_df['event_idx_within_trial']==eind-1)]
                    e_fix_centroid_prev = e_df_temp_prev.iloc[0]['fixation-centroid']
                    x_prev = e_fix_centroid_prev[0]
                    y_prev = e_fix_centroid_prev[1]
                    print("sacc dir angle")
                    # add angle
                    dummy_df.loc[e_df_inds, 'saccade-dir-angle'] = math.atan2((y-y_prev), (x-x_prev))+math.pi # make between 0-360

            # is first macrosaccade?
            dummy_df['is-first-macrosaccade'] = dummy_df['shape-macrosaccade-index']==1
            # bin saccade direction into 4 quadrants
            dummy_df['saccade-dir-angle-bin'] = pd.cut(dummy_df['saccade-dir-angle'], bins=4, labels=["quadrant1", "quadrant2", "quadrant3", "quadrant4"])
            print("concatting all dataframes...")
            dummy_df_all = pd.concat([dummy_df_all, dummy_df], ignore_index=True)
        # finally, set SP.DfScalar to dummy_df
        print("setting dfscalar...")
        #dummy_df_all = dummy_df_all.reset_index(drop=True)
        # filter outliers out
        if filter_outliers==True:
            print("filter outliers")
            dummy_df_all = dummy_df_all[dummy_df_all["shape-fixation"]!='FAR_FROM_ALL_SHAPES'].reset_index(drop=True)
        if filter_only_first_shapefix==True:
            print("filter only first shapefix)")
            dummy_df_all = dummy_df_all[dummy_df_all["first-fixation-on-shape"]==True].reset_index(drop=True)
        print("length of old dfscalar: ", len(self.DfScalar))
        self.DfScalar = dummy_df_all
        print("length of new dfscalar: ", len(self.DfScalar))
        print("dfscalar columns: ", self.DfScalar.columns)

    def load_v2(self, savedir):
        """ To load data saved using save_v2
        NOTE: Loads with outlier rows kept!
        """
        import pickle

        # remove recomputable things taht are large.
        # DfScalar = DfScalar.drop(["fr_sm_sqrt", "fr_sm_times", "fr_scalar_raw", "fr_scalar", "outlier_lims_upper", "outlier_remove"], axis=1)
        path = f"{savedir}/DfScalar.pkl"
        print("Loading: ", path)
        with open(path, "rb") as f:
            self.DfScalar = pickle.load(f)

        if "event" not in self.DfScalar.columns:
            self.DfScalar["event"] = self.DfScalar["event_aligned"]

        # save it
        # fr_sm_times = DfScalar.iloc[0]["fr_sm_times"]
        path = f"{savedir}/fr_sm_times.pkl"
        print("Loading: ", path)
        with open(path, "rb") as f:
            fr_sm_times = pickle.load(f)
        # Deal to each row
        tmp = [fr_sm_times for _ in range(len(self.DfScalar))]
        self.DfScalar["fr_sm_times"] = tmp

        # Other things
        attr_save = ["DS", "Params", "ParamsGlobals", "Sites", "Trials"]
        for a in attr_save:
            path = f"{savedir}/{a}.pkl"
            print("Loading: ", path)
            with open(path, "rb") as f:
                this = pickle.load(f)
                setattr(self, a, this)

        # Explose sites. could be lowered due to pruning (e.g. low fr)
        self.Sites = sorted(self.DfScalar["chan"].unique().tolist())

        # Recompute scalar fr (to help remove outliers)
        if False: # takes too long. now does qquiock computation (not winodwed)
            print("Computing fr scalar ...")
            self.DfScalar = self.datamod_compute_fr_scalar(self.DfScalar)
        if False: # Skip, since have already removed outliers.
            print("Removing outliers (using fr scalar) ...")
            self.datamod_remove_outliers()

        # Adding just since other code expects it sometimes
        self.DfScalar["fr_sm_sqrt"] = self.DfScalar["fr_sm"]**0.5

        # Other stuff
        self.ListPA = None
        # self.DS = None
        self._LOADED = True
        if not hasattr(self, "DfScalar_OutlierRows"):
            # older (before around jan 2024) saved the entier dataset including outliers...
            # wihtout this field.
            self.DfScalar_OutlierRows = None
        else:
            self.datamod_append_outliers()

        ## CLEANUP
        if "event_aligned" in self.DfScalar.columns:
            self.DfScalar["event"] = self.DfScalar["event_aligned"]
            # del self.DfScalar["event_aligned"]

        # Sanity checks that sites line up with SN. This can be different if,
        # for example, SN now uses kilosort units, but SP was saved with TDT.
        sn = self.SN
        try:
            # all SP sites should be in sn.
            sites_sn = sn.sitegetterKS_map_region_to_sites_MULTREG()
            sites_both = [s for s in self.Sites if s in sites_sn]

            n_sites_sp = len(self.Sites)
            n_sites_sn = len(sites_sn)
            n_sites_both = len(sites_both)

            if "SPIKES_VERSION" in self.Params:
                assert self.Params["SPIKES_VERSION"] == sn.SPIKES_VERSION
            else:
                # infer whats the spikes version
                if (np.all(np.diff(self.Sites)>0)) & (self.Sites[-1] > 480):
                    # then this is tdt
                    assert sn.SPIKES_VERSION=="tdt"
                else:
                    assert sn.SPIKES_VERSION=="kilosort"


            assert n_sites_both/n_sites_sp>0.85, "Almost All sites in SP should be presnet in SN" # dont do 100% beucase Diego dlPFC
            assert n_sites_both/n_sites_sn>0.95, "Almost all sites in SN sholdk be in SP"

        except AssertionError as err:
            print(self.Params)
            print(self.Sites)
            print(sites_sn)
            print(sn.SPIKES_VERSION)
            print(n_sites_both, n_sites_sn, n_sites_sp)
            print("Sites that exist in SP, but not sn:", [s for s in self.Sites if s not in sites_sn])
            print("Sites that exist in sn, but not SP:", [s for s in sites_sn if s not in self.Sites])
            print("You should reextract Snippets or load MS with the same spikes version as in SP")
            print(err)
            raise NotEnoughDataException

    def _sanity_trial_and_chans_are_balanced(self, dfthis=None, trial_key="index_datapt"):
        """ Check that all channel have the same trials, and vice versa.
        Does this by checking that the num trials are matched across chans, so
        in rare cases this may fail (but very unlikely)
        RETURNS:
        - bool, True is good.
        """
        from pythonlib.tools.pandastools import grouping_count_n_samples_quick
        if dfthis is None:
            dfthis = self.DfScalar
        nmin, nmax = grouping_count_n_samples_quick(dfthis, ["chan", trial_key])
        if not nmin==nmax:
            print("Min and max n datapts across all conjunctions of trial and chan: ", nmin, nmax)
        return nmin==nmax

    def _sanity_fr_sm_times_identical(self):
        """ Check that fr_sm_times is the same across all rows
        RETURNS:
        - bool, False means some rows have different times ffrom each other.
        """

        if self._SanityFrSmTimesIdentical is None:
            # then do check
            x = np.concatenate(self.DfScalar["fr_sm_times"].tolist())
            self._SanityFrSmTimesIdentical = np.max(np.abs(np.diff(x, axis=0))) < 0.001

        return self._SanityFrSmTimesIdentical

    def check_if_single_or_mult_session(self):
        """
        Is this a Snip from a single session, or concatenated from multiple rec sessions?
        :return:
        - "mult" or "single"
        - sessions, list of ints.
        """
        if self._CONCATTED_SNIPPETS:
            assert self.SNmult is not None
            assert self.SN is None
            sessions = [sn.RecSession for sn in self.SNmult.SessionsList]
            return "mult", sessions
        else:
            assert self.SNmult is None
            assert self.SN is not None
            return "single", [self.SN.RecSession]

    def save_v2(self, savedir):
        """ To save, but instead of pruing then saving, as in save(), here
        extract only what needed. This is new version that doesnt use List PA.
        NOTES;
        - SN, ~2GB
        """
        import pickle

        if False:
            # Ideally keep all. can easily remove outliers.
            DfScalar = self.DfScalar_BeforeRemoveOutlier.copy()
        else:
            # Reconstruct pre-remove-outlier.
            DfScalar = self.datamod_append_outliers(return_copy=True)
            # DfScalar = self.DfScalar.copy()
            # DfScalar_OutlierRows = self.DfScalar_OutlierRows.copy()
            # DfScalar = pd.concat([DfScalar, DfScalar_OutlierRows]).reset_index(drop=True)

        # 2) fr_sm_times
        # check that all have same fr times
        assert self._sanity_fr_sm_times_identical(), "trials have different times..."
        # x = np.concatenate(DfScalar["fr_sm_times"].tolist())
        # assert np.max(np.abs(np.diff(x, axis=0))) < 0.001, "trials have different times..."

        # save it
        fr_sm_times = DfScalar.iloc[0]["fr_sm_times"]
        path = f"{savedir}/fr_sm_times.pkl"
        print("SAving: ", path)
        with open(path, "wb") as f:
            pickle.dump(fr_sm_times, f)

        # remove recomputable things taht are large.
        list_cols_drop = ["fr_sm_times", "fr_scalar_raw", "fr_scalar", "outlier_lims_upper", "outlier_remove"]
        if "fr_sm_sqrt" in DfScalar.columns:
            list_cols_drop.append("fr_sm_sqrt")
        DfScalar = DfScalar.drop(list_cols_drop, axis=1)
        path = f"{savedir}/DfScalar.pkl"
        print("SAving: ", path)
        with open(path, "wb") as f:
            pickle.dump(DfScalar, f)

        # Other things
        attr_save = ["DS", "Params", "ParamsGlobals", "Sites", "Trials"]
        for a in attr_save:
            path = f"{savedir}/{a}.pkl"
            this = getattr(self, a)

            print("SAving: ", path)
            with open(path, "wb") as f:
                pickle.dump(this, f)    

        # Save yaml files of params
        from pythonlib.tools.expttools import writeDictToYaml
        path = f"{savedir}/Params.yaml"
        writeDictToYaml(self.Params, path)

        path = f"{savedir}/ParamsGlobals.yaml"
        writeDictToYaml(self.ParamsGlobals, path)

        path = f"{savedir}/trials_sites.yaml"
        dict_trials_sites = {
            "trials":self.Trials,
            "sites":self.Sites
        }
        writeDictToYaml(dict_trials_sites, path)

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
            if hasattr(self.SN, attr):
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
                if attr in store_attr.keys():
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
            if attr in store_attr.keys():
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

    def prune_low_firing_rate_sites(self, thresh = 1.5, PLOT=False):
        """ Removes Sites with fr lower than thresh. Does so simply by removing them from
         self.Sites, and not removing all the data from self.DfScalar """

        # Get FR mean for each site
        map_site_to_fr = {}
        for site in self.Sites:
            frmat, times, dict_plot_vals = self.dataextract_as_frmat(site)
            frmean = np.mean(frmat)
            map_site_to_fr[site] = frmean

        # WHich sites keep?
        sites_keep = [s for s,fr in map_site_to_fr.items() if fr>thresh]
        print("Keeping this many sites that pass fr thresh:")
        print(len(sites_keep), "/", len(map_site_to_fr))
        print("Using threshold: ", thresh)
        print("Updated self.Sites")

        # Update
        self.Sites = sites_keep

        # Plot
        if PLOT:
            fig, ax = plt.subplots()
            ax.hist(map_site_to_fr.values(), bins=100)
            ax.axvline(thresh, color="r")
        else:
            fig = None

        return fig


    def debug_subsample_prune(self, n_chans=32, n_trials=20):
        """ Prune dataset for quicker analyses. Do this by 
        subsampling chans and trials, in order to try to maintain balanced
        dataset
        RETURNS:
        - modifies slef.DfScalar, with reseted indices.
        NOTE: saves a copy of orig data in self.DfScalarBeforePrune
        """
        from pythonlib.tools.listtools import random_inds_uniformly_distributed

        if self.DfScalarBeforePrune is not None:
            # use the original data
            self.DfScalar = self.DfScalarBeforePrune
        
        print("Len of orig data: ", len(self.DfScalar))

        # Prune SP
        chans = sorted(self.DfScalar["chan"].unique().tolist())
        trialcodes = sorted(self.DfScalar["trialcode"].unique().tolist())

        chans_get = [chans[i] for i in random_inds_uniformly_distributed(chans, n_chans)]
        trialcodes_get = [trialcodes[i] for i in random_inds_uniformly_distributed(trialcodes, n_trials)]

        # store backup copy of DfScalar
        self.DfScalarBeforePrune = self.DfScalar.copy()

        print(chans_get)
        print(trialcodes_get)

        # replace with subsample
        self.DfScalar = self.DfScalar[(self.DfScalar["chan"].isin(chans_get)) & (self.DfScalar["trialcode"].isin(trialcodes_get))].reset_index(drop=True)

        print("Len of pruned data: ", len(self.DfScalar))
        self.Sites = sorted(self.DfScalar["chan"].unique().tolist())
        self.Trials = None



def extraction_helper(SN, which_level="trial", list_features_modulation_append=None,
    dataset_pruned_for_trial_analysis = None, NEW_VERSION=True, PRE_DUR = -0.6,
    POST_DUR = 0.6, PRE_DUR_FIXCUE=None, remove_low_fr_sites=True,
                      EVENTS_KEEP=None):
    """ Helper to extract Snippets for this session
    PARAMS;
    - list_features_modulation_append, eitehr None (do nothing) or list of str,
    which are features to add for computing modulation for.
    - PRE_DUR_FIXCUE, if not None, this overwrites predur for fixcue, which is useful becuase
     it is onset of trial, might wnat to make shorter.
    """

    # General cleanup of dataset.
    if False:
        if dataset_pruned_for_trial_analysis is None:
            # This is generally what you want
            dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(SN)
    else:
        # Dont exclude ANY data. DO this AFTER extracting Snippets.
        dataset_pruned_for_trial_analysis = SN.Datasetbeh.copy()

    if which_level=="trial":
        # Events
        do_prune_by_inter_event_times = False
        dict_events_bounds, dict_int_bounds = SN.eventsanaly_helper_pre_postdur_for_analysis(
            do_prune_by_inter_event_times=do_prune_by_inter_event_times,
            just_get_list_events=True)

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
        prune_feature_levels_min_n_trials = 1

        if PRE_DUR_FIXCUE:
            for i, (ev, pre, post) in enumerate(zip(list_events, list_pre_dur, list_post_dur)):
                if ev=="fixcue":
                    list_pre_dur[i] = PRE_DUR_FIXCUE
            print(f"Updated predur of fixcue to {PRE_DUR_FIXCUE}. New events and durs:")
            print(list_events)
            print(list_pre_dur)
            print(list_post_dur)

        trials_prune_just_those_including_events = True # True is fine, this just checkl that has fix touch
        DS_pruned = None
        fail_if_times_outside_existing = True

    elif which_level in ["stroke", "stroke_off", "substroke", "substroke_off"]:
        # Extracts snippets aligned to strokes. Features are columns in DS.
        list_events = [] # must be empty
        list_features_extraction = [] #
        list_features_get_conjunction = []
        # also those for computing moduulation.
        list_pre_dur = [PRE_DUR]
        list_post_dur = [POST_DUR]
        strokes_only_keep_single = False # if True, then prunes dataset, removing trials "remove_if_multiple_behstrokes_per_taskstroke"
        prune_feature_levels_min_n_trials = 1
        dataset_pruned_for_trial_analysis = None
        trials_prune_just_those_including_events = False
        fail_if_times_outside_existing = True
        DS_pruned = None

    elif which_level == "saccade_fix_on": # relevant for dfallpa_extraction_load_wrapper
        which_level="flex"
        # Extracts snippets aligned to strokes. Features are columns in DS.
        list_events = ["fixon_preparation"]
        list_features_extraction = [] #
        list_features_get_conjunction = []
        # also those for computing moduulation.
        list_pre_dur = [-0.5]
        list_post_dur = [0.5]
        strokes_only_keep_single = False
        tasks_only_keep_these=None
        prune_feature_levels_min_n_trials = None
        dataset_pruned_for_trial_analysis = None
        trials_prune_just_those_including_events = False
        fail_if_times_outside_existing=False
        fr_which_version='sqrt'
        SKIP_DATA_EXTRACTION=False,
        DS_pruned = None

    else:
        print(which_level)
        assert False

    if list_features_modulation_append is not None:
        assert isinstance(list_features_modulation_append, list)
        list_features_extraction = list_features_extraction + list_features_modulation_append
        list_features_get_conjunction = list_features_get_conjunction + list_features_modulation_append

    #### Subsample events
    if EVENTS_KEEP is not None:
        # Firest, remove numbers, e.g,, ['03_samp', 'go_cue'] --> ['samp', 'go_cue']
        tmp = []
        for ev in EVENTS_KEEP:
            try:
                int(ev[:2])
                a = True
            except ValueError as err:
                a = False
            except Exception as err:
                raise err
            if a==True and ev[2]=="_":
                # Then is like "03_samp"
                tmp.append(ev[3:])
            else:
                tmp.append(ev)
        EVENTS_KEEP = tmp
        # Second, keep only desired events.
        inds_keep = [i for i, ev in enumerate(list_events) if ev in EVENTS_KEEP]
        list_events = [list_events[i] for i in inds_keep]
        if len(list_pre_dur)>1:
            list_pre_dur = [list_pre_dur[i] for i in inds_keep]
            list_post_dur = [list_post_dur[i] for i in inds_keep]

    print("Kept these events: ", list_events)

    SP = Snippets(SN,
        which_level,
        list_events,
        list_features_extraction,
        list_features_get_conjunction,
        list_pre_dur,
        list_post_dur,
        strokes_only_keep_single=False,
        tasks_only_keep_these=None,
        prune_feature_levels_min_n_trials=prune_feature_levels_min_n_trials,
        dataset_pruned_for_trial_analysis=dataset_pruned_for_trial_analysis,
        trials_prune_just_those_including_events=trials_prune_just_those_including_events,
        fr_which_version='sqrt',
        SKIP_DATA_EXTRACTION=False,
        fail_if_times_outside_existing=fail_if_times_outside_existing,
        DS_pruned=DS_pruned)

    # Prune, to remove low FR sites
    if remove_low_fr_sites:
        SP.prune_low_firing_rate_sites()

    return SP

def datasetstrokes_extract(D, version, strokes_only_keep_single=False, tasks_only_keep_these=None,
    prune_feature_levels_min_n_trials=None, list_features=None, vel_onset_twindow = (0, 0.2)):
    """ Helper to extract dataset strokes
    PARAMS:
    - strokes_only_keep_single, bool, if True, then prunes dataset: 
    "remove_if_multiple_behstrokes_per_taskstroke"
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes, preprocess_dataset_to_datstrokes

    if not isinstance(version, str):
        print(version)
        assert False

    if list_features is None:
        list_features = []

    # 1. Extract all strokes, as bag of strokes.
    if False:
        DS = DatStrokes(D)
    else:
        # For PIG, singleprims, etc.
        # DS = preprocess_dataset_to_datstrokes(D, version="clean_one_to_one")
        DS = preprocess_dataset_to_datstrokes(D, version=version)

    # for features you want, if they are not in DS, then try extracting from D
    for feat in list_features:
        if feat not in DS.Dat.columns:
            print("Extracting from D.Dat --> DS.Dat:", feat)
            DS.dataset_append_column(feat)
    # OLD 
    # for f in list_features:
    #     assert f in DS.Dat.columns, "must extract this feature first. is it a datseg feature that you failed to extract?"

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

    # list_features = ["task_kind", "gridsize", "shape_oriented", "gridloc"]
    for key in list_features:
        print(" ")
        print("--- Unique levels for this category: ", key)
        print(DS.Dat[key].value_counts())

    return DS

def dataset_extract_prune_general_dataset(D,
                                          list_superv_keep, preprocess_steps_append,
                                          remove_aborts, list_superv_keep_full):
                                          # list_superv_keep = None,
                                          # preprocess_steps_append=None, remove_aborts=True,
                                          # list_superv_keep_full=None):

    """ [Good] Helper to prune main dataset
    """

    Dcopy = D.copy()
    print("Starting length of D.Dat:", len(Dcopy.Dat))

    # Remove if aborted
    if remove_aborts:
        assert False, "do this with preprocessGood"
        Dcopy.filterPandas({"aborted":[False]}, "modify")
        print("Len, after remove aborts:", len(Dcopy.Dat))

    print("--BEFORE REMOVE; existing supervision_stage_concise:")
    print(Dcopy.Dat["supervision_stage_concise"].value_counts())
    if list_superv_keep == "all":
        # Then don't prune based on superv
        print("############ NOT PRUNING SUPERVISION TRIALS")
        pass
    elif list_superv_keep is None:
        print("############ TAKING ONLY NO SUPERVISION TRIALS")
        # list_superv_keep = LIST_SUPERV_NOT_TRAINING
        Dcopy.preprocessGood(params=["no_supervision"])
    else:
        print("############ TAKING ONLY THESE SUPERVISION TRIALS:")
        print(list_superv_keep)
        assert isinstance(list_superv_keep, list)
        assert False, "do this with preprocessGood"

        # Only during no-supervision blocks
        print("--BEFORE REMOVE; existing supervision_stage_concise:")
        print(Dcopy.Dat["supervision_stage_concise"].value_counts())
        print("Keeping only these supervision values: ", list_superv_keep)
        Dcopy.filterPandas({"supervision_stage_concise":list_superv_keep}, "modify")
        print("--AFTER REMOVE; existing supervision_stage_concise:")
        print(Dcopy.Dat["supervision_stage_concise"].value_counts())

    print("Dataset final len:", len(Dcopy.Dat))

    if list_superv_keep_full is not None:
        assert False, "do this with preprocessGood"
        print("--BEFORE REMOVE; existing supervision_stage_new:")
        print(Dcopy.Dat["supervision_stage_new"].value_counts())
        Dcopy.filterPandas({"supervision_stage_new":list_superv_keep_full}, "modify")
        print(Dcopy.Dat["supervision_stage_new"].value_counts())

    if preprocess_steps_append is not None:
        Dcopy.preprocessGood(params=preprocess_steps_append)
    return Dcopy
    

def _dataset_extract_prune_general(sn, list_superv_keep = None,
        preprocess_steps_append=None, remove_aborts=True,
        list_superv_keep_full=None):
    """
    Generic, should add to this.
    """
    return dataset_extract_prune_general_dataset(sn.Datasetbeh,
                                                 list_superv_keep = list_superv_keep,
                                                 preprocess_steps_append=preprocess_steps_append, remove_aborts=remove_aborts,
                                                 list_superv_keep_full=list_superv_keep_full)

def _dataset_extract_prune_sequence(sn, n_strok_max = 2):
    """ Prep beh dataset before extracting snippets.
    Add columns that are necessary and
    return pruned datsaet for sequence analysis.
    """
    assert False, "Use the general version"
    import pandas as pd
    D = sn.Datasetbeh

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
                dat[f"{i}_shape"] = "IGN"
                dat[f"{i}_loc"] = ("IGN", "IGN")

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

    assert False, "use GENERAL"
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
    assert False, "use D.preprocessGood(no superivsion) insetad"
    LIST_NO_SUPERV = LIST_SUPERV_NOT_TRAINING
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


