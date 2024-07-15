

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
    # LIST_COMBINE_AREAS = [False, True]
    LIST_COMBINE_AREAS = [True, False]
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    PLOT_EACH_IDX = False

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
        

        from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp_better
        savedir_base = f"{SAVEDIR}/better_seperate_for_each_morph_idx"
        os.makedirs(savedir_base, exist_ok=True)
        analy_psychoprim_score_postsamp_better(DFallpa, DSmorphsets, 
                                            map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                            map_tcmorphset_to_info,
                                            savedir_base, animal, date,
                                            PLOT_EACH_IDX=PLOT_EACH_IDX)
        
        ###########################
        from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp
        analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                            map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                            map_tcmorphset_to_info,
                                            SAVEDIR)
                
