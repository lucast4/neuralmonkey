"""
Load dataset (from server) and save it locally.

"""
from ..utils import monkeylogic as mkl
from ..classes.session import Session


def load_and_preprocess_single_session(date, rec_session, animal = "Pancho"):
    """ Load this single rec session, coping things to local from server and making
    some sanity check plots
    NOTE: returns None rec_session doesnt exist.
    """


    # ============= RUN
    expt = "*"
    beh_session = rec_session+1 # 1-indexing.
    sessdict = mkl.getSessionsList(animal, datelist=[date])
    beh_sess_list = [sess_expt[0] for sess_expt in sessdict[date]]

    if beh_session not in beh_sess_list:
        print(f"session {beh_session} doesnt exist in {beh_sess_list}")
        return

    if False:
        # get all sessions
        beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date]]
    else:
        # Get the single session assued to map onto this neural.
        beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date] if sess_expt[0]==beh_session]
        assert(len(beh_expt_list))==1, "must be error, multiple sessions with same session num"
        beh_sess_list = [beh_session]

    beh_trial_map_list = [(1, 0)]
    sites_garbage = None

    print("Loading these beh expts:", beh_expt_list)
    print("Loading these beh sessions:",beh_sess_list)
    print("Loading this neural session:", rec_session)

    SN = Session(date, beh_expt_list, beh_sess_list, beh_trial_map_list, sites_garbage=sites_garbage,
                rec_session = rec_session, do_all_copy_to_local=True)

    # Load and save data
    SN.extract_raw_and_spikes_helper()

    print("** COMPLETED load_and_save_locally !!")

if __name__=="__main__":
    import sys

    date = sys.argv[1]

    # ============== PARAMS
    for rec_session in range(10):
        # go thru many, if doesnt exist will not do it.
        # rec_session = 1 # assumes one-to-one mapping between neural and beh sessions.
        load_and_preprocess_single_session(date, rec_session)

    # # ============== PARAMS
    # list_date = [220702, 220703, 220630, 220628, 220624, 220616, 220603, 220609]
    # for date in list_date:
    #     for rec_session in range(10):
    #         # go thru many, if doesnt exist will not do it.
    #         # rec_session = 1 # assumes one-to-one mapping between neural and beh sessions.
    #         load_and_preprocess_single_session(date, rec_session)

