

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



if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    # WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    classifier_ver = "logistic"
    # classifier_ver = "ensemble"

    which_level = "trial"
    question = "CHAR_BASE_trial"
    # events_keep = ["03_samp", "06_on_strokeidx_0"]
    events_keep = None
    # LIST_COMBINE_AREAS = [False, True]
    LIST_COMBINE_AREAS = [True, False]
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    for COMBINE_AREAS in LIST_COMBINE_AREAS:
        
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/charparsing/{animal}-{date}-{classifier_ver}-combine={COMBINE_AREAS}"
        os.makedirs(SAVEDIR, exist_ok=True)
        
        #####################################
        # Method 2 - Combine two dfallpa
        DFallpa1 = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS)
        DFallpa2 = load_handsaved_wrapper(animal=animal, date=date, version="stroke", combine_areas=COMBINE_AREAS)
        DFallpa = pd.concat([DFallpa1, DFallpa2]).reset_index(drop=True)

        #################### PREPROCESSING
        from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
        dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

        # dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT=False)

        # for pa in DFallpa["pa"]:
        #     pa.X = pa.X**0.5
            
        # from neuralmonkey.classes.population_mult import dfpa_concat_normalize_fr_split_multbregion_flex
        # fr_mean_subtract_method = "across_time_bins"
        # # fr_mean_subtract_method = "each_time_bin"
        # PLOT=False

        # dfpa_concat_normalize_fr_split_multbregion_flex(DFallpa, fr_mean_subtract_method, PLOT)

        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
        shape_var_suff = "shapesemgrp"
        loc_var_suff = "loc_on_clust"
        dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

        ###########################
        from neuralmonkey.analyses.decode_moment import analy_chars_score_postsamp, analy_chars_dfscores_condition
        analy_chars_score_postsamp(DFallpa, SAVEDIR, animal, date) 