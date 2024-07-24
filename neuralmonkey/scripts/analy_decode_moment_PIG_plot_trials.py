"""
Plot single trials of moment by moment decoding -- takes a while.
Written for PIG and shapesequnece (using PIG code).

"""

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

from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper

if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    do_syntax_rule_stuff = int(sys.argv[3])==1
    # WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    TBIN_DUR = 0.15

    classifier_ver = "logistic"
    # classifier_ver = "ensemble"

    # which_level = "trial"
    # question = "CHAR_BASE_trial"
    # events_keep = ["03_samp", "06_on_strokeidx_0"]
    # events_keep = None
    # LIST_COMBINE_AREAS = [False, True]

    if animal == "Diego":
        LIST_COMBINE_AREAS = [True]
        LIST_BREGION = ["PMv", "PMd"]
    elif animal == "Pancho":
        LIST_COMBINE_AREAS = [False]
        LIST_BREGION = ["PMv_m", "PMd_p", "PMd_a"]
    
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]
    for COMBINE_AREAS in LIST_COMBINE_AREAS:
        
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/SINGLE_TRIAL_PLOTS/{animal}-{date}-{classifier_ver}-combine={COMBINE_AREAS}-syntax={do_syntax_rule_stuff}"
        os.makedirs(SAVEDIR, exist_ok=True)
        
        #####################################
        # Method 2 - Combine two dfallpa
        try:
            DFallpa1 = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS)
            DFallpa2 = load_handsaved_wrapper(animal=animal, date=date, version="stroke", combine_areas=COMBINE_AREAS)
        except Exception as err:
            DFallpa1 = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS, use_time=False)
            DFallpa2 = load_handsaved_wrapper(animal=animal, date=date, version="stroke", combine_areas=COMBINE_AREAS, use_time=False)
        DFallpa = pd.concat([DFallpa1, DFallpa2]).reset_index(drop=True)

        #################### PREPROCESSING
        dfpa_concatbregion_preprocess_wrapper(DFallpa)

        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
        shape_var_suff = "shape"
        loc_var_suff = "loc"
        dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

        ### Preprocess - Get PCs, for plotting
        list_papca = []
        for i, row in DFallpa.iterrows():
            twind = row["twind"]
            pa = row["pa"]
            print(row["event"], row["bregion"])
            X, PAfinal, PAslice, pca, X_before_dimred = pa.dataextract_state_space_decode_flex(twind_overall=twind,
                                                    tbin_dur=0.15, tbin_slide=0.02,
                                                    reshape_method = "chans_x_trials_x_times",
                                                    pca_reduce=True, npcs_keep_force=10)
            assert pa.X.shape[1] == PAfinal.X.shape[1]
            list_papca.append(PAfinal)
        DFallpa["pa_pca"] = list_papca

        ### Extract beh data
        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import prepare_beh_dataset
        MS, D, shape_sequence, map_tc_to_syntaxconcrete, map_trialcode_loc_to_shape = prepare_beh_dataset(animal, 
                                                                                                        date, 
                                                                                                        do_syntax_rule_stuff=do_syntax_rule_stuff,
                                                                                                        return_MS=True)        

        # Get eye trackign, if not already got
        for sn in MS.SessionsList:
            if not sn.clusterfix_check_if_preprocessing_complete():
                sn.extract_and_save_clusterfix_results()
        
        ########################### PLOTS

        from neuralmonkey.analyses.decode_moment import plot_single_trial_combine_signals_wrapper

        # LIST_TRAIN_DATA = ["sp_samp", "strokes"]
        LIST_TRAIN_DATA = ["sp_samp"]

        ####
        # LIST_BREGION = DFallpa["bregion"].tolist()
        if do_syntax_rule_stuff:
            color_by = "shape_order_global"
            syntax_shape_sequence = shape_sequence
        else:
            color_by = "stroke_index_seqc"
            syntax_shape_sequence = None
            
        for bregion in LIST_BREGION:
            # pull out trialcodes to plot
            # For now, just plot those where he drew each stroke.
            pa = extract_single_pa(DFallpa, bregion, None, "trial", "03_samp")
            dflab = pa.Xlabels["trials"]
            TRIALCODES_PLOT = dflab[(dflab["FEAT_num_strokes_beh"] == dflab["FEAT_num_strokes_task"])]["trialcode"].tolist()
            for train_dataset_name in LIST_TRAIN_DATA: 
                plot_single_trial_combine_signals_wrapper(DFallpa, bregion, TRIALCODES_PLOT, train_dataset_name, 
                                                            SAVEDIR, MS, color_by=color_by, 
                                                            syntax_shape_sequence=syntax_shape_sequence,
                                                            TBIN_DUR=TBIN_DUR)
                
            plt.close("all")