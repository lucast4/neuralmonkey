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

        assert len(list_sessions)>0
        
        self.SessionsList = list_sessions
        self.Datasetbeh = None
        self._generate_index() 


    ################ PARAMS
    def animal(self):
        """ return the common animal
        if different animals, then error
        """
        animals = list(set([sn.Animal for sn in self.SessionsList]))
        assert len(animals)==1
        return animals[0]


    def date(self):
        """ return the common date
        if different dates, then error
        """

        dates = list(set([sn.Date for sn in self.SessionsList]))
        assert len(dates)==1
        return dates[0]


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

    # def index_convert(self, index, output="sn_trial"):
    def index_convert_trial_trialcode_flex(self, index, output="sn_trial"):
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
            sessnum, trial_in_sess = index
        else:
            print(index)
            assert False, "code it"

        SN = self.SessionsList[sessnum]
        return SN, trial_in_sess, sessnum

    def session_extract_largest(self):
        """ Get the sn with the most trials"""
        # pull out the largest sn
        list_n = []
        for i, sn in enumerate(self.SessionsList):
            list_n.append(len(sn.Datasetbeh.Dat))
        i = list_n.index(max(list_n))
        sn = self.SessionsList[i]
        return sn


    ########### DATASET beh
    def datasetbeh_extract(self):
        """ Get concatenated dataset, 
        First time will extract and cache.
        RETURNS:
        - D, concatted dataset
        """
        from pythonlib.dataset.analy_dlist import concatDatasets

        if self.Datasetbeh is None:
            # Load it for one time.

            if len(self.SessionsList)==1:
                # Just use the one.
                D = self.SessionsList[0].Datasetbeh
            else:
                from pythonlib.dataset.dataset import load_dataset_daily_helper

                # Get for this day.
                # Better than concatting -- load from scratch, can't trust concatDatasets, sometimes for cases
                # where use prims labeled by cluster, it changes...
                # Merge datasets
                D = load_dataset_daily_helper(self.animal(), self.date())

                # prune to just the trialcodes that exist
                trialcodes = []
                for sn in self.SessionsList:
                    trialcodes.extend(sn.Datasetbeh.Dat["trialcode"].unique().tolist())
                trialcodes = list(set(trialcodes))

                D.Dat = D.Dat[D.Dat["trialcode"].isin(trialcodes)].reset_index(drop=True)

                # Dlist = []
                # for sn in self.SessionsList:
                #     Dlist.append(sn.Datasetbeh)
                # D = concatDatasets(Dlist)

            self.Datasetbeh = D

        return self.Datasetbeh

    #################### UTILS
    def prune_remove_sessions_too_few_trials(self, min_n_trials):
        """ for each suessions removes if trials (clean) 
        (doesnt care whetehr exists in beh dataset) 
        not enough.
        RETURNS:
        - modifies self.SessionsList in place
        """
        def _good(sn):
            return len(sn.get_trials_list(True, True, False))>=min_n_trials

        print("[MS] prune_remove_sessions_too_few_trials...")
        print("Starting  num sessions: ", len(self.SessionsList))
        self.SessionsList = [sn for sn in self.SessionsList if _good(sn)]
        print("Ending num sessions: ", len(self.SessionsList))


    # def session_generate_single_nodata(self):
    #     """ Generate a single sn for use in analyses that aggregated across
    #     sessions, where sn is used for thigns that are common across
    #     sessions, such as site info, and for methods.
    #     Takes the first session, and clears all the neural data, and
    #     does sanity check that all sessions have same metadata
    #     """
        

    ################### SITES
    # (Generally asserts that all sessions have same channels ...)
    def sitegetter_all(self, list_regions=None, clean=True, how_combine="assert_identical"):
        """ Gets list of sites. Runs this for each sessin and checsk that 
        all are identicayul before returning
        PARAMS:
        - how_combine, if sessions have diff sites, how to combine? 
        """

        list_list_sites = []
        for SN in self.SessionsList:
            list_sites = SN.sitegetterKS_map_region_to_sites_MULTREG(list_regions, clean)
            list_list_sites.append(list_sites)

        if how_combine=="assert_identical":
            # check that all lists are same
            for i, sites1 in enumerate(list_list_sites):
                for ii, sites2 in enumerate(list_list_sites):
                    if ii>i:
                        if set(sites1)!=set(sites2):
                            print(len(sites1), sites1)
                            print(len(sites2), sites2)
                            assert False

            return list_list_sites[0]
        elif how_combine=="intersect":
            # get the intersection
            sites_intersect = list_list_sites[0]
            for sites_this in list_list_sites[1:]:
                sites_intersect = [s for s in sites_intersect if s in sites_this]
            return sites_intersect
        elif how_combine=="union":
            assert False, "code it"
        else:
            print(how_combine)
            assert False

    def print_summary_sessions(self):
        """ Helper to print summary of n trial and sites for each sn
        """
        print("=== N trials per session")
        for i, sn in enumerate(self.SessionsList):
            print("sess", i, len(sn.get_trials_list(True)))

        print("=== N units per session")
        for i, sn in enumerate(self.SessionsList):
            print("\n====== SESSION NUM: ", i)
            sn.sitegetter_print_summary_nunits_by_region()
