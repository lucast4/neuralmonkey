""" A collection of sessions, 
Does not touch the raw data (therfore Session is the interface between this processed
and raw data)
"""

class MultSessions(object):
    """docstring for MultSessions"""
    def __init__(self, list_sessions):
        """
        PARAMS:
        - list_sessions, list of Session objects.
        """

        self.SessionsList = list_sessions

        self._generate_index() 



    ################ Indexing
    def _generate_index(self):
        """ run once to generate indexes
        """

        # Map from trialcode to session
        self._MapperTrialcode2SessTrial = {}
        self._MapperInd2Trialcode = {}
        ind = 0
        for sessnum, SN in enumerate(self.SessionsList):
            for trialcode, trial in SN._MapperTrialcode2TrialToTrial.items():
                self._MapperTrialcode2SessTrial[trialcode] = (sessnum, trial)
                self._MapperInd2Trialcode[ind] = trialcode
                ind+=1
        print("Generated index mappers!")

    def index_convert(self, index, output="sn_trial"):
        """ 
        Flexible, helps convert multiple possible ways of indexing trials into 
        Session object and trial within that Session.
        PARAMS:
        - index, different possibilites:
        --- ind, int counts from 0 to N, for N trials, in temporal order
        --- (sess num, trial_within_sess), tuple
        --- trialcode, str, date-<beh_session>-<beh_trial>
        RETURN:
        - Depends on output:
        --- [if output=="sn_trial"], Session, trial in sess, sessnum
        --- [if output=="trialcode"], trialcode
        """

        if isinstance(index, int):
            # incremental index over entire lsti of sessions.
            trialcode = self._MapperInd2Trialcode[index]
            sessnum, trial_in_sess = self._MapperTrialcode2SessTrial[trialcode]
        elif isinstance(index, str):
            # trialcode
            sessnum, trial_in_sess = self._MapperTrialcode2SessTrial[index]
        elif isinstance(index, tuple):
            # (sessnum trial)
            assert len(index)==2
            sessnum, trial = index
        else:
            print(index)
            assert False, "code it"

        SN = self.SessionsList[sessnum]
        return SN, trial_in_sess, sessnum




    ################### SITES
    # (Generally asserts that all sessions have same channels ...)
    def sitegetter_all(self, list_regions=None, clean=True):
        """ Gets list of sites. Runs this for each sessin and checsk that 
        all are identicayul before returning
        """

        list_list_sites = []
        for SN in self.SessionsList:
            list_sites = SN.sitegetter_all(list_regions, clean)
            list_list_sites.append(list_sites)

        # check that all lists are same
        for i, sites1 in enumerate(list_list_sites):
            for ii, sites2 in enumerate(list_list_sites):
                if ii>i:
                    if set(sites1)!=set(sites2):
                        print(len(sites1), sites1)
                        print(len(sites2), sites2)
                        assert False

        return list_list_sites[0]
