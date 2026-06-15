"""
Specifically to make plots of heatmaps of shape vs. location, for single prims,
good, for paper.

NOTEBOOK: 241110_shape_invariance_all_plots_SP.ipynb

"""

import sys
import os

from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED

SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN/MULT"

if __name__=="__main__":

    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import mult_load_euclidian_time_resolved
    from neuralmonkey.scripts.analy_euclidian_chars_sp_MULT import plot_scalar_all, plot_timecourse_all

    animal = sys.argv[1]
    date = int(sys.argv[2])
    var_other = sys.argv[3] # seqc_0_loc or gridsize

    # LIST_ANIMAL_DATE_COMB_VAROTHER = [
    #     ("Diego", 230615, True, "seqc_0_loc"), 
    # ]
    
    LIST_ANIMAL_DATE_COMB_VAROTHER = [
        (animal, date, True, var_other), 
    ]
    
    DFDIST = mult_load_euclidian_time_resolved(LIST_ANIMAL_DATE_COMB_VAROTHER)

    # Make save directory
    a = "_".join(set([x[0] for x in LIST_ANIMAL_DATE_COMB_VAROTHER]))
    b = min([x[1] for x in LIST_ANIMAL_DATE_COMB_VAROTHER])
    c = max([x[1] for x in LIST_ANIMAL_DATE_COMB_VAROTHER])
    var_other = [str(x[3]) for x in LIST_ANIMAL_DATE_COMB_VAROTHER][0]
    SAVEDIR = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}-varother={var_other}"
    print(SAVEDIR)
    os.makedirs(SAVEDIR, exist_ok=True)

    ### SCALAR SUMMARY
    savedir = f"{SAVEDIR}/SCALAR"
    os.makedirs(savedir, exist_ok=True)
    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.3), (0.3, 0.6), (0.05, 0.6), (0.5, 1.0)],
            "05_first_raise":[(-0.5,  -0.1), (-0.1, 0.5)],
            "06_on_strokeidx_0":[(-0.5, -0.1), (0, 0.5)],
        }
    plot_scalar_all(DFDIST, savedir, map_event_to_listtwind, var1="seqc_0_shape", var2=var_other, reverse_axis_order=True)

    ### TIMECOURSE SUMMARY
    plot_timecourse_all(DFDIST, SAVEDIR, var1="seqc_0_shape", var2=var_other)


