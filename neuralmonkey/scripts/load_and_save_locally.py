"""
Load dataset (from server) and save it locally.

"""
from ..utils import monkeylogic as mkl
from ..classes.session import Session, load_session_helper
import os
import matplotlib.pyplot as plt

DATES_IGNORE = ["220708"]

def load_and_preprocess_single_session(date, rec_session, animal = "Pancho"):
    """ Load this single rec session, coping things to local from server and making
    some sanity check plots
    NOTE: returns None rec_session doesnt exist.
    PARAMS:
    - date, str.
    """
    from ..utils.monkeylogic import session_map_from_rec_to_ml2
    from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles

    dataset_beh_expt = None
    expt = "*"

    print(" in load_and_preprocess_single_session")

    if date in DATES_IGNORE:
        print("*** SKIPPING DATE (becasue it is in DATES_IGNORE): ", date)

    # ============= RUN
    # beh_session = rec_session+1 # 1-indexing.
    out = session_map_from_rec_to_ml2(animal, date, rec_session)
    if out is None:
        # Then this rec/beh session doesnt exist
        # (Otherwise, continue)
        print("DIDNT FIND THIS SESSIONS", animal, date, rec_session)
        return

    # else:


    # sessdict = mkl.getSessionsList(animal, datelist=[date])

    # print("ALL SESSIONS: ")
    # print(sessdict)

    # if all([len(x)==0 for x in sessdict.values()]):
    #     # skip, this animal and date doesnt exits.
    #     return sessdict

    # # beh_sess_list = [sess_expt[0] for sess_expt in sessdict[date]]

    # # Not enough beh sessions?
    # if len(sessdict[date])<rec_session+1:
    #     # Then skip this rec_session, there is not a matching beh sessinon
    #     print(f"rec session {rec_session} doesnt exist (no matching beh session found)")
    #     return

    try:
        SN = load_session_helper(date, dataset_beh_expt, rec_session, animal, expt, 
            do_all_copy_to_local=True)
    except Exception as err:
        print("&&&&&&&&&&&&")
        print(date, dataset_beh_expt, rec_session, animal, expt)
        raise err

    # if beh_session not in beh_sess_list:
    #     print(f"session {beh_session} doesnt exist in {beh_sess_list}")
    #     return

    # # Get the single session assued to map onto this neural.
    # beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date] if sess_expt[0]==beh_session]
    # assert(len(beh_expt_list))==1, "must be error, multiple sessions with same session num"
    # beh_sess_list = [beh_session]

    # beh_trial_map_list = [(1, 0)]
    # sites_garbage = None

    # print("Loading these beh expts:", beh_expt_list)
    # print("Loading these beh sessions:",beh_sess_list)
    # print("Loading this neural session:", rec_session)

    # SN = Session(date, beh_expt_list, beh_sess_list, beh_trial_map_list, sites_garbage=sites_garbage,
    #             rec_session = rec_session, do_all_copy_to_local=True)


    ########## EXTRACT THINGS TO SAVE LOCALLY.
    # Load and save data
    SN.extract_raw_and_spikes_helper()

    # Spike trains:
    print("** [load_and_preprocess_single_session] Step 1: spiketrain_as_elephant_batch")
    SN.spiketrain_as_elephant_batch()

    # Check beh code and photodiode match
    if False:
        # Skip for now, need to fix:
        # at this line, gets error that list index out of range: FLOOR = x["valminmax"][0]
        print("** [load_and_preprocess_single_session] Step 2: plot_behcode_photodiode_sanity_check")
        SN.plot_behcode_photodiode_sanity_check()

    # Save
    print("** [load_and_preprocess_single_session] Step 3: plot_spike_waveform_multchans")
    path = f"{SN.Paths['figs_local']}/waveforms_overlay"
    if not checkIfDirExistsAndHasFiles(path)[1]:
    # if not os.path.exists(path):
        SN.plot_spike_waveform_multchans(LIST_YLIM = [[-250, 100]])

    print("** [load_and_preprocess_single_session] Step 4: plot_spike_waveform_stats_multchans")
    path = f"{SN.Paths['figs_local']}/waveforms_stats"
    if not checkIfDirExistsAndHasFiles(path)[1]:
    # if not os.path.exists(path):
        SN.plot_spike_waveform_stats_multchans(None)

    # Get stats about fr
    print("** [load_and_preprocess_single_session] Step 5: sitestats_fr_get_and_save")
    SN.sitestats_fr_get_and_save()

    # Save stats about fr (usefeul for pruning low fr sitse and nosy sites)
    print("** [load_and_preprocess_single_session] Step 6: plotbatch_sitestats_fr_overview")
    SN.plotbatch_sitestats_fr_overview()

    # Save (1) sanity check and (2) cached extraction of photodiode crossings, for
    # timing of events
    SN.events_get_time_using_photodiode_and_save()
    path = f"{SN.Paths['figs_local']}/eventcodes_trial_structure"
    if not checkIfDirExistsAndHasFiles(path)[1]:
        SN.eventsdataframe_sanity_check()
    
    print("** COMPLETED load_and_save_locally !!")

if __name__=="__main__":
    import sys

    date = sys.argv[1]

    # ============== PARAMS
    for rec_session in range(10):
        # go thru many, if doesnt exist will not do it.
        # rec_session = 1 # assumes one-to-one mapping between neural and beh sessions.
        print("Running:", sys.argv, "session: ", rec_session)
        load_and_preprocess_single_session(date, rec_session)
        plt.close("all")

    # # ============== PARAMS
    # list_date = [220702, 220703, 220630, 220628, 220624, 220616, 220603, 220609]
    # for date in list_date:
    #     for rec_session in range(10):
    #         # go thru many, if doesnt exist will not do it.
    #         # rec_session = 1 # assumes one-to-one mapping between neural and beh sessions.
    #         load_and_preprocess_single_session(date, rec_session)

