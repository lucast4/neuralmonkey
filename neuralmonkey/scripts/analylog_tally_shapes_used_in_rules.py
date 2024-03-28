"""
Across many days, save summary of the shapes used in rules expts, including their chunk rank, and which epochs.

For each day, prints what shapes used in each chunk, and plots drawings of examples.
Across days, in a single text file, prints all the days.

3/25/24 - Goal: to find days that use different shapes but same chunk rank and rules.
"""

# %cd ..
# from tools.utils import *
# from tools.plots import *
# from tools.analy import *
# from tools.calc import *
# from tools.analyplot import *
# from tools.preprocess import *
# from tools.dayanalysis import *

from pythonlib.drawmodel.analysis import *
from pythonlib.tools.stroketools import *
import pythonlib
from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper
import pickle
import seaborn as sns
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES



RES = []

if __name__=="__main__":
    import sys
    import os

    animal = sys.argv[1]
    ANALY_VER = sys.argv[2]

    # NOTE on these dates: Spans all clumps of dates, so that this should capture all shapes that were trained.
    # For rulesw, did not get all cases with >2 rules. Should get those if this doesnt show much variation.
    if animal=="Pancho" and ANALY_VER=="rulesingle":
        # DATES = ["220901", "220908", "220909", "230810", "230826"] # analyzed
        DATES = ["220831", "220901", "220902", "220906", "220907", "220908", "220909", "230810", "230811", "230824",
                 "230826", "230829", "231114", "231116"] # ALL that exist (gspreadsheet)
    elif animal=="Pancho" and ANALY_VER=="rulesw":
        # DATES = ["221020", "230905", "230910", "230914", "230919"] # analyzed
        DATES = ["221020", "230905", "230910", "230911", "230912", "230914", "230918", "230919", "230928",
                 "230929", "221019", "221020", "221024", "221021", "221023"] # ALMOST ALL of (shapes vs dir, shapes vs color, dir dir shape)
    elif animal=="Diego" and ANALY_VER=="rulesingle":
        DATES = ["230724", "230726", "230817", "230913", "230730", "231118"] # ALMOST ALL that exist (gspreadsheet)
    elif animal=="Diego" and ANALY_VER=="rulesw":
        DATES = ["230917", "230823", "230804", "230809", "230825", "230813", "230827", "230919",
                 "230901", "230905", "230910", "230907", "230912", "231001",
                 "230703", "230705", "230711", "230713", "230719"] # ALMOST ALL of (shapes vs dir, shapes vs color, dir dir shape)
    else:
        print(animal, ANALY_VER)
        assert False, "inputs incorrect"

    SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/TALLIES/shapes_used_in_rules/{ANALY_VER}-{animal}"
    os.makedirs(SAVEDIR, exist_ok=True)

    for date in DATES:
        ######## LOAD DATASET
        D = load_dataset_daily_helper(animal, date)

        ######## PREPROCESS, FEATURES.
        from neuralmonkey.metadat.analy.anova_params import dataset_apply_params
        DS = None
        D, DS, params = dataset_apply_params(D, DS, ANALY_VER, animal, date) # prune it

        ######### PRINT SHAPES USED.
        from pythonlib.dataset.dataset_strokes import DatStrokes
        from pythonlib.tools.pandastools import grouping_print_n_samples
        DS = DatStrokes(D)
        DS.dataset_append_column("epochset")
        savedir = f"{SAVEDIR}/{date}"
        os.makedirs(savedir, exist_ok=True)

        LIST_VARS = [
            ["epochset", "epoch", "chunk_rank", "chunk_within_rank", "shape"],
            ["epoch", "chunk_rank", "shape"],
            ["shape", "epoch", "chunk_rank"],
            ["epoch", "chunk_rank", "chunk_within_rank","shape"]
            ]
        for vars in LIST_VARS:
            grouping_print_n_samples(DS.Dat, vars, save_as="txt", savepath=f"{savedir}/{'|'.join(vars)}.txt")

        # Also plot examples of shapes.
        figbeh, figtask = DS.plotshape_row_col_vs_othervar("epoch", n_examples_per_sublot=3)
        from pythonlib.tools.plottools import savefig
        savefig(figbeh, f"{savedir}/shape_drawings-BEH.pdf")
        if figtask is not None:
            savefig(figbeh, f"{savedir}/shape_drawings-TASK.pdf")

        ########## Collect information - (epoch, chunk_rank, shape)
        outdict = grouping_print_n_samples(DS.Dat, ["epoch", "chunk_rank", "shape"], save_as="txt")

        RES.append({
            "ANALY_VER":ANALY_VER,
            "animal":animal,
            "date":date,
            "outdict":outdict
        })

    ######### A single text file holding all results.
    savepath = f"{SAVEDIR}/overview.txt"
    lines = []
    for this in RES:
        lines.append((this["ANALY_VER"], this["animal"], this["date"]))
        lines.extend([f"  {str(k)} : {v}" for k, v in this["outdict"].items()])

    from pythonlib.tools.expttools import writeStringsToFile
    writeStringsToFile(savepath, lines)