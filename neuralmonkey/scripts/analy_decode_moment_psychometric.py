

import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa



def prune_dfscores_good_morphset(dfscores, morphset_get, animal, date):
    """
    """

    print("morphset_get:", morphset_get)
    if morphset_get == "good_ones":
        # Hand modified
        if (animal, date) == ("Diego", 240515):
            # Angle rotation
            morphsets_ignore = [2] # Did most with two strokes...
        if (animal, date) == ("Diego", 240523):
            # THis is a garbage morphset, is not actually morphing.
            morphsets_ignore = [0]
        elif (animal, date) == ("Pancho", 240521):
            morphsets_ignore = [0] # one base prim is garbage
        elif (animal, date) == ("Pancho", 240524):
            morphsets_ignore = [4] # doesnt actually vaciallte across tirals
        else:
            morphsets_ignore = []
    elif morphset_get is None:
        # Get all morphsets
        morphsets_ignore = []
    elif isinstance(morphset_get, int):
        # get just this one morphset (exclude the others)
        morphsets_ignore = [ms for ms in DF_TCRES["run_morphset"].unique().tolist() if ms!=morphset_get]
    else:
        print(morphset_get)
        assert False

    # Prune dfscores
    morphsets_keep = [ms for ms in dfscores["run_morph_set_idx"].unique().tolist() if ms not in morphsets_ignore]
    print("morphsets_ignore: ", morphsets_ignore)
    print("morphsets_exist: ", sorted(dfscores["run_morph_set_idx"].unique()))
    print("morphsets_keep: ", morphsets_keep)
    dfscores = dfscores[dfscores["run_morph_set_idx"].isin(morphsets_keep)].reset_index(drop=True)
    print("morphsets_exist (after pruen): ", sorted(dfscores["run_morph_set_idx"].unique()))

    return dfscores


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    # WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    classifier_ver = "logistic"
    # classifier_ver = "ensemble"

    which_level = "trial"
    # LIST_COMBINE_AREAS = [False, True]
    LIST_COMBINE_AREAS = [True, False]
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    PLOT_EACH_IDX = False

    PLOTS_DO = [1]

    for COMBINE_AREAS in LIST_COMBINE_AREAS:
        
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP/{animal}-{date}-{classifier_ver}-combine={COMBINE_AREAS}"
        os.makedirs(SAVEDIR, exist_ok=True)

        #####################################
        # Method 2 - Combine two dfallpa
        DFallpa = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS)

        #################### PREPROCESSING
        from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
        dfpa_concatbregion_preprocess_wrapper(DFallpa)


        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
        shape_var_suff = "shape"
        loc_var_suff = "loc"
        dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

        from neuralmonkey.analyses.decode_moment import analy_psychoprim_prepare_beh_dataset
        savedir = f"{SAVEDIR}/morphsets_drawings"
        os.makedirs(savedir, exist_ok=True)
        DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, \
            map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = analy_psychoprim_prepare_beh_dataset(animal, date, savedir)
        
        # Prune neural data to keep only those trialcodes that are in DSmorphsets
        list_trialcode = DSmorphsets.Dat["trialcode"].unique().tolist()
        DFallpa["pa"] = [pa.slice_by_labels("trials", "trialcode", list_trialcode) for pa in DFallpa["pa"].values]

        if 0 in PLOTS_DO:
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp_better
            savedir_base = f"{SAVEDIR}/better_seperate_for_each_morph_idx"
            os.makedirs(savedir_base, exist_ok=True)
            analy_psychoprim_score_postsamp_better(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                savedir_base, animal, date,
                                                PLOT_EACH_IDX=PLOT_EACH_IDX)
        
        ###########################
        if 1 in PLOTS_DO:
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp
            analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                SAVEDIR,
                                                animal=animal, date=date)
                
