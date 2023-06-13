""" Use this for cases where a rec_session overlaps multiple beh sessions
"""

def load_beh_trial_map_list(animal, DATE, rec_session):
    """ One rec session -- Multiple beh sessions. Here hard
    code the mapping. Actually, can just run 
    session_map_from_rec_to_ml2_ntrials_mapping to auto get mapping,
    as long as there is only one rec session that day...
    """
    if animal=="Pancho" and int(DATE)==221024 and rec_session==1:
        # two rec sessions, one beh session
        beh_trial_map_list = [(667, 0)]
    elif animal=="Pancho" and int(DATE)==220719:
        # one rec session, two beh sessions
        beh_trial_map_list = [(1,0), (1,45)]
    else:
        # Default, perfect alignment.
        beh_trial_map_list = [(1, 0)]

    return beh_trial_map_list
