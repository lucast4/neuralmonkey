""" for converting between beh, neural, and camrera
"""

def get_map_trial_and_set(mapping_list, ntrials):
    """ get dict mapping between trial (set2) and 
    setnum and trial1 (set 1). given a maping list.
    PARAMS:
    - mapping_list, list of mappings, where each mapping 
    is (trial1, trial2). Trial2 must be incresaing, (usualyl meaning
    that it did not crash in data collection). Asserts that
    NOTE:
    - useful e.g., for nerual and beh, where if beh (set 1)has 2 session
    to one neural (set 2), mapping list is:
    [(1,0), (1, 34)], if trial 34 is when started new ml2. 
    e.g., mapping_list = [(1, 0), (2, 5), (2, 7)] -->
        {0: (0, 1),
         1: (0, 2),
         2: (0, 3),
         3: (0, 4),
         4: (0, 5),
         5: (1, 2),
         6: (1, 3),
         7: (2, 2),
         8: (2, 3),
         9: (2, 4),
         10: (2, 5),
         11: (2, 6),
         12: (2, 7),
         13: (2, 8),
    """

    def convert_trial_(mapping, trial2):
        """
        Convert between two sets, trial numbers
        mapping = [20, 0] means that trial 20 for set 1 is trial 0 for set 2
        trialright = 5 means get me the trial for set 1 if trial for set 2 is 5
        """
        indthis = trial2-mapping[1]
        trial1 = indthis + mapping[0]
        return trial1

    # trial2 nums must be monotonic incresaing.
    prev = mapping_list[0][1]
    for m in mapping_list[1:]:
        if m[1]<=prev:
            print(mapping_list)
            assert False, "trial2 must be increasing."
        prev = m[1]
        
    onsets = [x[1] for x in mapping_list] # the first trial (set 2) for each set 1 set
    dict_trial2_to_set_and_trial1 = {}
    for trial2 in range(ntrials):
        setnum = sum([trial2>=o for o in onsets])-1
        mapping = mapping_list[setnum]
        trial1 = convert_trial_(mapping, trial2)
        dict_trial2_to_set_and_trial1[trial2] = (setnum, trial1)

    return dict_trial2_to_set_and_trial1
