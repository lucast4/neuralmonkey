from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper

# ALL (ALL DONE)
# LIST_DATE = ["220608", "220616", "220624", "220630", "220714", "220715", "220730", "220805", "220816", 
#     "220827", "221002", "221020", "221107", "221125", "220106", "220109", "220126", "220310", "230320"]

assert False, "run load_and_save_locally instead. which does al preprocessing steps, skipping those that are already done"

LIST_DATE = ["220908"]

animal = "Pancho"
dataset_beh_expt = None

for DATE in LIST_DATE:
    
    MS = load_mult_session_helper(DATE, animal, MINIMAL_LOADING=False)
    # for sn in MS.SessionsList:
    #     sn.datasetbeh_load_helper(dataset_beh_expt)

    for sn in MS.SessionsList:
        sn._savelocalcached_extract()
        sn._savelocalcached_save(save_datslices=True)
