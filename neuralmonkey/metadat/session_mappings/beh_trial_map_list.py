""" Use this for cases where a rec_session overlaps multiple beh sessions

NOTES:
- mult rec sessions, one beh session. 
---> To do, write a variant of session_map_from_rec_to_ml2_ntrials_mapping(), which
deals with Mult rec, one beh. Currently it is difciult as need to know how many rec
trials exist. To do so, solutionw ould be to load each rec session, count its n trials, 
use that to dcide one hwne next trial starts, run local preprocessing, and then move up
one session. 
OR have a way to extract tdt bank beh codes, trail onsets and offsets) to determine n trials.
Currently even bare_bones loeading doesnt work. that doesnt have the paths needed to load tank.
Approach might be to modify the barebones loading.
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
    - NOTE: this is NOT actually used for anything (except checing if output is None or IGNORE)... it is done automaticlaly downstream.
    """

    if True: # Since I dont think it is doing antyhing, this is sanity, it should work
        if animal=="Pancho" and int(DATE)==220719 and rec_session>0:
            # one rec session, two beh sessions
            beh_trial_map_list = "IGNORE"
        else:
            # Default, perfect alignment.
            beh_trial_map_list = None
    else:
        if animal=="Pancho" and int(DATE)==221024 and rec_session==1:
            # two rec sessions, one beh session
            beh_trial_map_list = [(667, 0)]
        elif animal=="Pancho" and int(DATE)==220719 and rec_session==0:
            # one rec session, two beh sessions
            beh_trial_map_list = [(1,0), (1,45)]
        elif animal=="Pancho" and int(DATE)==220719 and rec_session>0:
            # one rec session, two beh sessions
            beh_trial_map_list = "IGNORE"
        elif animal=="Pancho" and int(DATE)==220610 and rec_session==0:
            # Thrw out first beh session with 40 trials, but kept rec on. 
            # # Therefore, throw out first 40 rec trials
            beh_trial_map_list = [(1, 42)]
        # elif animal=="Pancho" and int(DATE)==220710 and rec_session==1:
        #     # This 2nd rec session should start earlier in beh session
        #     beh_trial_map_list = (229, 0)
        elif animal=="Pancho" and int(DATE)==220609 and rec_session==0:
            # Thrw out first few 10 neural trials, for some reason...
            beh_trial_map_list = [(1, 10)]
        else:
            # Default, perfect alignment.
            beh_trial_map_list = None

    if beh_trial_map_list is not None and not beh_trial_map_list=="IGNORE":
        assert isinstance(beh_trial_map_list, list)
        assert isinstance(beh_trial_map_list[0], tuple)

    return beh_trial_map_list
