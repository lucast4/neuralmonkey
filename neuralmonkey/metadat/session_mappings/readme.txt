Different scenarios.

Rec and beh sessions shifted (at level of session, not trial) (but same N and 1 to 1):
- input into rec_to_beh_Diego.yaml

Mult beh session, one rec session (no spillover)
- run session_map_from_rec_to_ml2_ntrials_mapping()

Mult rec fit completely into one beh session (no spillover)
- NOT DONE
---> To do, write a variant of session_map_from_rec_to_ml2_ntrials_mapping(), which
deals with Mult rec, one beh. Currently it is difciult as need to know how many rec
trials exist. To do so, solutionw ould be to load each rec session, count its n trials, 
use that to dcide one hwne next trial starts, run local preprocessing, and then move up
one session. 
OR have a way to extract tdt bank beh codes, trail onsets and offsets) to determine n trials.
Currently even bare_bones loeading doesnt work. that doesnt have the paths needed to load tank.
Approach might be to modify the barebones loading.

Corrupted, lost beh trials
- Input date in corrupted_ml2_sessions.yaml. 
- Then hacky input the missed trials in get_trials_list(). Search for 231206. (see corrupted...yaml for how to determine what the trials are.)

Corrupted, lost neural trials
- (e.g., Pancho 220614, I think the first neural trial should be thrown out)
- is currently very hackily entered in get_trials_list()
- ??
