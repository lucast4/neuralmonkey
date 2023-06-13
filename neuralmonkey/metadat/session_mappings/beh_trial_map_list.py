""" Use this for cases where a rec_session overlaps multiple beh sessions
"""

def load_beh_trial_map_list(animal, DATE, rec_session):
    """ One rec session -- Multiple beh sessions. Here hard
    code the mapping. Actually, can just run 
    session_map_from_rec_to_ml2_ntrials_mapping to auto get mapping,
    as long as there is only one rec session that day...
    RETURNS:
    - beh_trial_map_list, either:
    --- list of 2-tuples. see within. this means this rec session exists, and
    must be mapped this way
    --- None, have no info
    --- "IGNORE", then you shouild ignore this rec session. important for cases where
    woulnd['t know that previuos rec sessions used up all the beh sesions.
    NOTE: 
    - you must always use rec_session and DATE, or else will return wrong beh_trial_map_list
    """
    if animal=="Pancho" and int(DATE)==221024 and rec_session==1:
        # two rec sessions, one beh session
        beh_trial_map_list = [(667, 0)]
    elif animal=="Pancho" and int(DATE)==220719 and rec_session==0:
        # one rec session, two beh sessions
        beh_trial_map_list = [(1,0), (1,45)]
    elif animal=="Pancho" and int(DATE)==220719 and rec_session>0:
        # one rec session, two beh sessions
        beh_trial_map_list = "IGNORE"
    else:
        # Default, perfect alignment.
        beh_trial_map_list = None

    return beh_trial_map_list
