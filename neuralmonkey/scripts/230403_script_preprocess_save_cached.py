from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper


LIST_DATE = ["220616", "220624", "220630", "220714", "220827", "221020", "221107", "230320"]
animal = "Pancho"
dataset_beh_expt = None

for DATE in LIST_DATE:
    
    MS = load_mult_session_helper(DATE, animal)
    # for sn in MS.SessionsList:
    #     sn.datasetbeh_load_helper(dataset_beh_expt)

    for sn in MS.SessionsList:
        sn._savelocalcached_extract()
        sn._savelocalcached_save(save_datslices=True)
