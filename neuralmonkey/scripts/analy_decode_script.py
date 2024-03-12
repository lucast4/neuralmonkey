"""
Good general-purpose script for all decoding stuff, and plotting of decode.
(No state space plots)
Builds on analyquick_decode_substrokes, but there was quick and hacky, here is after
started library of good decoding functions (decode_good.py). Can look back at analyquick_decode_substrokes, ight ahve other stuff.

Older code: analyquick_decode_substrokes
Notebook: 240217_snippets_decode_all

LOG:
2/27/24 - Good state.

2/20/24 - Good state.
Approach is to copy to 240217_snippets_decode_all to develop, but this should be assumed to hold
latest.
DONE:
- cross-temporal, use kfold, but only for same-event comparisons, since across events
runs into issue where trials are not identical. --> TODO: fix this.
- good pruning of data for conjunctiion analyses before passing into the decode_good.py
functions, which should clean things up.
- Sure that the labels are matching across all analyses, even for shape variable nakmes that
are diff (e.g,, seqc_0_shape, seqc_1_shape), by digitizing data BEFORE anything.
TODO:
- Decode sequence conditions on current context and other vraibles. This is done in
notebook (see # 2b) Separate decoder for each level of other var (then take average over decoders). Useful to controlling for variables
)
- Single-trial decoding. This is working fine, but not consolidated (see
### Extract single trial results
"""

from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels

SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/DECODE"
DEBUG = False



if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])
    do_combine = int(sys.argv[3]) # 0, 1

    if do_combine == 1:
        # COMBINE trial and stroke
        dir_suffix = sys.argv[4]
        question = None
        q_params = None
        combine_trial_and_stroke = True
        which_level = None
        q_params = {
            "effect_vars": ["seqc_0_shape", "seqc_0_loc"]
        }
        if dir_suffix=="PIG":
            question_trial = "PIG_BASE_trial"
            question_stroke = "PIG_BASE_stroke"
        elif dir_suffix=="CHAR":
            question_trial = "CHAR_BASE_trial"
            question_stroke = "CHAR_BASE_stroke"
        else:
            print(dir_suffix)
            assert False

        # Since chars have small N per shape, do clumping.
        HACK_RENAME_SHAPES = "CHAR" in question_trial
    else:
        # DONT COMBINE, use questions.
        question = sys.argv[4]
        combine_trial_and_stroke = False
        which_level = sys.argv[5]
        # which_level = "stroke" # Hand code for now...
        dir_suffix = question

        # Load q_params
        from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
        q_params = rsagood_questions_dict(animal, date, question)[question]
        if which_level=="trial":
            events_keep = ["03_samp", "04_go_cue", "06_on_strokeidx_0"]
        elif which_level=="stroke":
            events_keep = ["00_stroke"]
        else:
            assert False

        HACK_RENAME_SHAPES = "CHAR" in question

    plots_do = str(sys.argv[6]) # 1111 means do all 4 plots. 1101 means skip 3rd.
    assert len(plots_do)==4
    PLOT_1 = plots_do[0]=="1"
    PLOT_2 = plots_do[1]=="1"
    PLOT_3 = plots_do[2]=="1"
    PLOT_4 = plots_do[3]=="1"
    # else:
    #     PLOT_1, PLOT_2, PLOT_3, PLOT_4 = True, True, True, True

    ############### PARAMS
    exclude_bad_areas = True
    SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
    combine_into_larger_areas = False
    list_time_windows = [(-0.6, 0.6)]
    N_MIN_TRIALS = 5
    # EVENTS_IGNORE = ["04_go_cue"] # To reduce plots
    EVENTS_IGNORE = [] # To reduce plots
    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/DECODE/{animal}-{date}/{dir_suffix}-combined_{do_combine}-combine_areas_{combine_into_larger_areas}"
    os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

    if question=="SP_novel_shape":
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=True
    else:
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=False

    ############################ SAVE PARAMS
    writeDictToTxt({
        "events_keep":events_keep,
        "question":question,
        "q_params":q_params,
        "combine_trial_and_stroke":combine_trial_and_stroke,
        "which_level":which_level,
        "HACK_RENAME_SHAPES":HACK_RENAME_SHAPES,
        "exclude_bad_areas":exclude_bad_areas,
        "SPIKES_VERSION":SPIKES_VERSION,
        "combine_into_larger_areas":combine_into_larger_areas,
        "list_time_windows":list_time_windows,
        "N_MIN_TRIALS":N_MIN_TRIALS}, f"{SAVEDIR_ANALYSIS}/params.txt")

    ########################################## RUN
    if combine_trial_and_stroke:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper_combine_trial_strokes
        DFallpa = dfallpa_extraction_load_wrapper_combine_trial_strokes(animal, date, question_trial,
                                                                           question_stroke,
                                                    list_time_windows,
                                                   combine_into_larger_areas = combine_into_larger_areas,
                                                   exclude_bad_areas=exclude_bad_areas,
                                                    SPIKES_VERSION=SPIKES_VERSION,
                                                    HACK_RENAME_SHAPES = HACK_RENAME_SHAPES,
                                                   do_fr_normalization=True)
    else:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
        DFallpa = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows,
                                                  which_level=which_level, events_keep=events_keep,
                                                  combine_into_larger_areas = combine_into_larger_areas,
                                                  exclude_bad_areas = exclude_bad_areas,
                                                  SPIKES_VERSION = SPIKES_VERSION,
                                                  HACK_RENAME_SHAPES = HACK_RENAME_SHAPES,
                                                  do_fr_normalization=True)

    #################### HACKY, for wl=="trial", extract motor and other variables for first stroke.
    from neuralmonkey.classes.population_mult import dfallpa_preprocess_vars_conjunctions_extract
    dfallpa_preprocess_vars_conjunctions_extract(DFallpa, which_level=which_level)

    # PREPROCESS - factorize all relevant labels FIRST here.
    from neuralmonkey.analyses.decode_good import preprocess_factorize_class_labels_ints
    MAP_LABELS_TO_INT = preprocess_factorize_class_labels_ints(DFallpa, savepath = f"{SAVEDIR_ANALYSIS}/MAP_LABELS_TO_INT.txt")

    # Figure out how long is seuqence
    if which_level=="trial":
        pa = DFallpa["pa"].values[0]
        n_strokes_max = -1
        for i in range(6):
            if f"seqc_{i}_shape" in pa.Xlabels["trials"].columns:
                n_ignore = sum(pa.Xlabels["trials"][f"seqc_{i}_shape"].isin(["IGN", "IGNORE"]))
                n_total = len(pa.Xlabels["trials"][f"seqc_{i}_shape"])
                print(n_ignore, n_total)
                if n_ignore<n_total:
                    n_strokes_max=i+1
        assert n_strokes_max>0
    else:
        pa = DFallpa["pa"].values[0]
        n_strokes_max = max(pa.Xlabels["trials"]["stroke_index"])+1
    print("THIS MANY STROKES MAX:", n_strokes_max)

    ## Novel prims -- hacky, to test decode separately for novel vs. learned prims, turn this flag on, to make task_kind
    # into conjunctive string <task_kind>|<shape_is_novel>
    if TASK_KIND_RENAME_AS_NOVEL_SHAPE:
        from pythonlib.tools.pandastools import append_col_with_grp_index
        list_panorm = []
        for pa in DFallpa["pa"].tolist():
            if "task_kind_orig" not in pa.Xlabels["trials"].columns:
                pa.Xlabels["trials"]["task_kind_orig"] = pa.Xlabels["trials"]["task_kind"]
                pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], ["task_kind_orig", "shape_is_novel_all"], "task_kind")
            else:
                pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], ["task_kind_orig", "shape_is_novel_all"], "task_kind", use_strings=True, strings_compact=True)

    ################################ PLOTS
    list_br = DFallpa["bregion"].unique().tolist()
    list_tw = DFallpa["twind"].unique().tolist()
    list_ev = DFallpa["event"].unique().tolist()

    # Prune events, to reduce size of plots.
    list_ev = [ev for ev in list_ev if ev not in EVENTS_IGNORE]

    ################################# VARIABLES
    # Sequence prediction stuff
    LIST_VAR_DECODE = []
    LIST_VARS_CONJ = []
    LIST_SEPARATE_BY_TASK_KIND = []
    LIST_FILTDICT = []
    LIST_SUFFIX = []

    # SETS OF HARD CODED VARIABLES.
    # if combine_trial_and_stroke==False and question in ["CHAR_BASE_stroke", "PIG_BASE_stroke"]:
    from neuralmonkey.metadat.analy.anova_params import params_getter_decode_vars
    if combine_trial_and_stroke==False and which_level=="trial":
        # For sequence-related decoding (and also location and shape).
        # Should work for SP, PIG, and CHAR
        # FOCUS ON THINGS DURING PLANNING (stroke-related stuff do with stroke plots).

        # Just focus on 1D decoding, and do "combine" if want to do 2d.
        PLOT_1 = False
        PLOT_2 = True
        PLOT_3 = False
        PLOT_4 = False

        LIST_VAR_DECODE, LIST_VARS_CONJ, LIST_SEPARATE_BY_TASK_KIND, LIST_FILTDICT, LIST_SUFFIX = params_getter_decode_vars(which_level)

    elif combine_trial_and_stroke==False and which_level=="stroke":
        # For sequence-related decoding (and also location and shape).
        # Should work for SP, PIG, and CHAR

        PLOT_1 = False
        PLOT_2 = True
        PLOT_3 = False
        PLOT_4 = False

        LIST_VAR_DECODE, LIST_VARS_CONJ, LIST_SEPARATE_BY_TASK_KIND, LIST_FILTDICT, LIST_SUFFIX = params_getter_decode_vars(which_level)


    ############## 1) Default: Time-resolved decoding
    if PLOT_1:
        SAVEDIR = f"{SAVEDIR_ANALYSIS}/1_time_resolved"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)

        from neuralmonkey.utils.frmat import bin_frmat_in_time
        from neuralmonkey.analyses.decode_good import decodewrap_categorical_timeresolved_singlevar, decodewrapouterloop_categorical_timeresolved
        from pythonlib.tools.plottools import savefig
        import pandas as pd
        RES = []

        n_strokes_max_this = min([n_strokes_max, 4])
        list_var_decode = [f"seqc_{i}_shape" for i in range(n_strokes_max_this)]
        if "gridloc" in q_params["effect_vars"] or "seqc_0_loc" in q_params["effect_vars"]:
            list_var_decode += [f"seqc_{i}_loc" for i in range(n_strokes_max_this)]
        if "gridsize" in q_params["effect_vars"]:
            list_var_decode += ["gridsize"]

        DFRES = decodewrapouterloop_categorical_timeresolved(DFallpa, list_var_decode,
                                                     SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                     n_min_trials=N_MIN_TRIALS)


    ################### 4) Cross- and Within- condition decoding
    from neuralmonkey.analyses.decode_good import decodewrapouterloop_categorical_timeresolved_cross_condition, decodewrapouterloop_categorical_timeresolved_within_condition
    if PLOT_2:

        # PARAMS
        subtract_mean_vars_conj = False # WHether to normalize by sutbracting mean within each level of othervar...

        if len(LIST_VAR_DECODE)==0:
            ##### CROSS-CONDITION
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/2_cross_decode"
            os.makedirs(SAVEDIR, exist_ok=True)
            print(SAVEDIR)

            SKIP=False
            if "gridloc" in q_params["effect_vars"] or "seqc_0_loc" in q_params["effect_vars"]:
                list_var_decode = ["shape_this_event", "loc_this_event"]
                list_vars_conj = [
                    ["loc_this_event"],
                    ["shape_this_event"]]
            elif "gridsize" in q_params["effect_vars"]:
                list_var_decode = ["shape_this_event", "size_this_event"]
                list_vars_conj = [
                    ["size_this_event"],
                    ["shape_this_event"]]
            else:
                SKIP=True

            if not SKIP:
                DFRES = decodewrapouterloop_categorical_timeresolved_cross_condition(DFallpa, list_var_decode,
                                                                     list_vars_conj,
                                                                     SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                                     subtract_mean_vars_conj=subtract_mean_vars_conj)

            ######### Within-condition decoding (and then average) -- i.e., this is control
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/2_within_decode"
            os.makedirs(SAVEDIR, exist_ok=True)
            print(SAVEDIR)

            # PARAMS
            DFRES = decodewrapouterloop_categorical_timeresolved_within_condition(DFallpa, list_var_decode,
                                                                 list_vars_conj,
                                                                SAVEDIR, time_bin_size=0.1, slide=0.05)
        else:
            for list_var_decode, list_vars_conj, separate_by_task_kind, filtdict, suffix in zip(
                    LIST_VAR_DECODE, LIST_VARS_CONJ, LIST_SEPARATE_BY_TASK_KIND, LIST_FILTDICT, LIST_SUFFIX):


                ##### CROSS-CONDITION
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/2_cross_decode/{suffix}"
                os.makedirs(SAVEDIR, exist_ok=True)
                print(SAVEDIR)
                writeDictToTxt({
                    "list_var_decode":list_var_decode,
                    "list_vars_conj":list_vars_conj,
                    "separate_by_task_kind":separate_by_task_kind,
                    "filtdict":filtdict,
                    "suffix":suffix}, f"{SAVEDIR}/params.txt")
                DFRES = decodewrapouterloop_categorical_timeresolved_cross_condition(DFallpa, list_var_decode,
                                                     list_vars_conj,
                                                     SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                     subtract_mean_vars_conj=subtract_mean_vars_conj,
                                                                                     filtdict=filtdict,
                                                                                     separate_by_task_kind=separate_by_task_kind)

                ######### Within-condition decoding (and then average) -- i.e., this is control
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/2_within_decode/{suffix}"
                os.makedirs(SAVEDIR, exist_ok=True)
                print(SAVEDIR)
                DFRES = decodewrapouterloop_categorical_timeresolved_within_condition(DFallpa, list_var_decode,
                                                                     list_vars_conj,
                                                                    SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                                                      filtdict=filtdict,
                                                                                     separate_by_task_kind=separate_by_task_kind)

    ################### # 4) Cross-decoding across time bins
    from neuralmonkey.analyses.decode_good import decodewrap_categorical_cross_time, decodewrapouterloop_categorical_cross_time_plot_compare_contexts, decodewrapouterloop_categorical_cross_time, decodewrapouterloop_categorical_cross_time_plot

    if PLOT_3:
        SAVEDIR = f"{SAVEDIR_ANALYSIS}/3_cross_temporal"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)

        # Make sure all pa use the same variables to refer to shapes.
        import warnings
        warnings.filterwarnings("default")

        # warnings.filterwarnings("error")
        time_bin_size = 0.1
        slide=0.05
        n_strokes_max_this = min([n_strokes_max, 3])
        # list_var_decode = [f"seqc_{i}_shape" for i in range(n_strokes_max_this)]
        list_var_decode = ["shape_this_event"]
        if n_strokes_max_this>1:
            # Just get max up to stroke 2, can run more if needed.
            list_var_decode += ["seqc_1_shape"]
        DFRES = decodewrapouterloop_categorical_cross_time(DFallpa, list_var_decode,
                                                           time_bin_size, slide, savedir=SAVEDIR)

        # -- PLOTS
        if len(DFRES)>0:
            decodewrapouterloop_categorical_cross_time_plot(DFRES, SAVEDIR)

            SAVEDIR = f"{SAVEDIR_ANALYSIS}/3_cross_temporal_split_by_contexts"
            os.makedirs(SAVEDIR, exist_ok=True)
            print(SAVEDIR)
            decodewrapouterloop_categorical_cross_time_plot_compare_contexts(DFRES, "shape_this_event", SAVEDIR)

    #########################################
    # 5) Cross-temporal decoder, but with decoder trained using one variable (e.g., seqc_0_shape) and testing prediction of another variable (e.g., seqc_1_shape).
    # Useful if, for example, want to see if decoder trained during visual presentation can decode 2nd stroke DURING 1st stroke.
    from neuralmonkey.analyses.decode_good import decodewrapouterloop_categorical_cross_time_cross_var, decodewrapouterloop_categorical_cross_time_plot
    if PLOT_4:
        if n_strokes_max>1:
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/4_cross_temporal_diff_var_train_test"
            os.makedirs(SAVEDIR, exist_ok=True)
            print(SAVEDIR)


            # Make sure all pa use the same variables to refer to shapes.
            time_bin_size = 0.1
            slide=0.05

            n_strokes_max_this = min([n_strokes_max, 3])

            # Decoder using seqc_0_shape
            list_var_decode = [f"seqc_{i}_shape" for i in range(1, n_strokes_max_this)]
            list_var_decode_train_test_0 = [["seqc_0_shape", sh] for sh in list_var_decode]  # (variable to construct decoder, variable that you will try to predict).

            # Decoder using seqc_1_shape
            list_var_decode = [f"seqc_{i}_shape" for i in range(2, n_strokes_max_this)]
            list_var_decode_train_test_1 = [["seqc_1_shape", sh] for sh in list_var_decode]  # (variable to construct decoder, variable that you will try to predict).

            list_var_decode_train_test = list_var_decode_train_test_0 + list_var_decode_train_test_1

            assert len(DFallpa["twind"].unique())==1, "not big deal. just change code below to iter over all (ev, tw)."

            DFRES = decodewrapouterloop_categorical_cross_time_cross_var(DFallpa,
                                                                 list_var_decode_train_test,
                                                                     time_bin_size, slide, savedir=SAVEDIR)
            if len(DFRES)>0:
                decodewrapouterloop_categorical_cross_time_plot(DFRES, SAVEDIR)

