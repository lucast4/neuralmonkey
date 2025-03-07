""" Working with pathsa nd directories
"""
import os.path
import sys

from pythonlib.globals import PATH_NEURALMONKEY, PATH_DATA_NEURAL_RAW, PATH_DATA_NEURAL_PREPROCESSED, PATH_KS_RAW, PATH_SAVE_CLUSTERFIX
from pythonlib.tools.expttools import writeStringsToFile, makeTimeStamp

from neuralmonkey.classes.session import LOCAL_LOADING_MODE, LOCAL_PATH_PREPROCESSED_DATA

def clusterfix_check_if_preprocessing_complete(animal, date, session_no=0):
    """
    Check whether clusterfix (eye tracking) has been done for this day.
    Assumes that if exists for session 0, then exists for all this date, and doesnt check contents of the folder, just that
    folder exists and has stuff
    RETURNS:
    - bool.
    """
    from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles    
    SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/{animal}-{date}-{session_no}/clusterfix_result_csvs"
    return checkIfDirExistsAndHasFiles(SAVEDIR)[1]

def find_ks_cluster_paths(animal, date):
    """
    Find paths, in list len sessions, to ks directiores (ei., duirng ks extraction, before concat across sessions for that day
    """
    from pythonlib.tools.expttools import findPath, deconstruct_filename

    path_hierarchy = [
        [animal],
        [date],
        [animal, date]
    ]

    paths = findPath(PATH_KS_RAW, path_hierarchy, path_fname=None, sort_by="name")

    # REmove paths that say "IGNORE"
    paths = [p for p in paths if "IGNORE" not in p]

    # OUTPUT AS DICT
    sessions = []
    for sessnum, paththis in enumerate(paths):
        fnparts = deconstruct_filename(paththis)
        final_dir_name = fnparts["filename_final_noext"]
        # print("---")
        # print(paththis)
        # print(fnparts)
        # print(final_dir_name)
        sessions.append({
            "sessnum":sessnum,
            "path":paththis,
            "pathfinal":final_dir_name,
            "fileparts":fnparts
            })

        # sessdict[sessnum] = []

    return sessions

def find_rec_session_paths(animal, date):
    """
    """
    from pythonlib.tools.expttools import findPath, deconstruct_filename

    path_hierarchy = [
        [animal],
        [date]
    ]

    if LOCAL_LOADING_MODE:
        paths = findPath(LOCAL_PATH_PREPROCESSED_DATA, path_hierarchy, sort_by="name")
    else:    
        paths = findPath(PATH_DATA_NEURAL_RAW, path_hierarchy, sort_by="name")

    # REmove paths that say "IGNORE"
    paths = [p for p in paths if "IGNORE" not in p]
    # # assert len(paths)==1, 'not yhet coded for combining sessions'
    # assert len(paths)>0, "maybe you didn't mount server?"
    # if len(paths)<self.RecSession+1:
    #     print("******")
    #     print(paths)
    #     print(self.RecSession)
    #     print(self.RecPathBase, path_hierarchy)
    #     print(self.Animal, self.Date)
    #     print(self.print_summarize_expt_params())
    #     assert False, "why mismatch?"

    # paththis = paths[self.RecSession]
    # # print(paths, self.RecSession)
    # # assert False

    # OUTPUT AS DICT
    sessions = []
    for sessnum, paththis in enumerate(paths):
        fnparts = deconstruct_filename(paththis)
        final_dir_name = fnparts["filename_final_noext"]
        # print("---")
        # print(paththis)
        # print(fnparts)
        # print(final_dir_name)
        sessions.append({
            "sessnum":sessnum,
            "path":paththis,
            "pathfinal":final_dir_name,
            "fileparts":fnparts
            })

        # sessdict[sessnum] = []

    return sessions


def check_log_preprocess_completion_status(animal, DEST_DIR= f"{PATH_NEURALMONKEY}/logs_checks"):
    """ Check preprocess status across all dates and sessions (gotten from raw/server)
    and wehther entire preprocess done (including cached), then write this info to file
    """
    from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED, PATH_DATA_NEURAL_RAW
    from pythonlib.tools.expttools import fileparts
    import glob
    from pythonlib.tools.expttools import writeStringsToFile, makeTimeStamp
    from neuralmonkey.classes.session import load_session_helper

    DEST_FILE = f"{DEST_DIR}/{animal}.txt"
    DEST_FILE_DONE_DATES = f"{DEST_DIR}/{animal}_done_dates.txt"

    # go thru raw data to collect dates and sessions
    DIR_RAW = f"{PATH_DATA_NEURAL_RAW}/{animal}"
    list_dir_date = glob.glob(f"{DIR_RAW}/*")

    res = []
    res_strings = []
    # dates_done = []
    dates_done_string = "" # " # a single flat string"
    for dirthis in list_dir_date:

        # List of sessiosn for this date
        list_dir_sessions = glob.glob(f"{dirthis}/*")
        # print(dirthis)
        print(dirthis)
        date = int(fileparts(dirthis)[-2])
        
        # count and check sessions
        date_string = f"{date}" # Initialize
        all_done = True
        for i, dirsess in enumerate(list_dir_sessions):
            
            sess_id = fileparts(dirsess)[-2] # e.g, Pancho-220715-154205
            DIR_PREPROCESS = f"{PATH_DATA_NEURAL_PREPROCESSED}/recordings/{animal}/{date}/{sess_id}"
            DIR_PREPROCESS_CACHED = f"{DIR_PREPROCESS}/cached"

            # check if preprocess done.
            try:
                SN = load_session_helper(date, None, i, animal, None,  
                    ACTUALLY_BAREBONES_LOADING = True) # Very quick loading
                SN.Paths = {
                    "cached_dir":DIR_PREPROCESS_CACHED
                }
                done = SN._savelocalcached_check_done(datslice_quick_check=True)
            except AssertionError as err:
                print(" ----- ", dirsess)
                print(err)
                continue
            
            # Save output
            if False:
                res.append({
                    "date":date,
                    "sessnum":i,
                    "done":done
                })
            
            # Append to string
            if done:
                date_string+=f"  {i}_done"
            else:
                date_string+=f"  ({sess_id})"
                all_done = False
            
        res_strings.append({
            "date":date,
            "string":date_string
        })
        
        # Store the dates that are done
        if all_done:
            # dates_done.append(date)
            dates_done_string = f"{dates_done_string} {date}"
                
    # write to file, after sorting by date
    res_strings = sorted(res_strings, key=lambda x: x["date"])
    list_strings = [x["string"] for x in res_strings]
    ts = makeTimeStamp()
    list_strings = [f"Checked on {ts}"] + ["date | sess-preprocess_done", "------------------------"] + list_strings
    writeStringsToFile(DEST_FILE, list_strings)

    # also save a text file with all done dates
    writeStringsToFile(DEST_FILE_DONE_DATES, [dates_done_string])
    print(f"Logged at {DEST_FILE}")


def check_log_anova_analy_status(DEST_DIR= f"{PATH_NEURALMONKEY}/logs_checks"):
    import glob
    import os

    DIR_BASE = "/gorilla1/analyses/recordings/main/anova"
    which_data = "bytrial"
    DIR_ANOVA = f"{DIR_BASE}/{which_data}/MULT_SESS"

    LIST_DIR_ANIMAL_DATE = sorted(glob.glob(f"{DIR_ANOVA}/*"))

    OUT_STRINGS = []
    for DIR_ANIMAL_DATE in LIST_DIR_ANIMAL_DATE:
        LIST_DIR_ANALYVER = sorted(glob.glob(f"{DIR_ANIMAL_DATE}/*"))
        
        OUT_STRINGS.append(" ") # add a space
        
        for DIR_ANALYVER in LIST_DIR_ANALYVER:
            
            LIST_DIR_VAR_VARSOTHER = sorted(glob.glob(f"{DIR_ANALYVER}/var_by_varsother/*"))
            
            # collect, which var-varsothers have data
            for DIR_VAR_VARSOTHER in LIST_DIR_VAR_VARSOTHER:
                
                LIST_DIR_SCOREVER = glob.glob(f"{DIR_VAR_VARSOTHER}/*")
                for DIR_SCOREVER in LIST_DIR_SCOREVER:
                    # print(DIR_SCOREVER)
                    
                    # Look for df_var in this directory
                    path_dfvar = f"{DIR_SCOREVER}/df_var.pkl"
                    df_var_exists = os.path.exists(path_dfvar)
                    
                    ####### GET METADATA
                    from pythonlib.tools.expttools import deconstruct_filename

                    ANIMAL = deconstruct_filename(DIR_ANIMAL_DATE)["filename_components_hyphened"][0]
                    DATE = deconstruct_filename(DIR_ANIMAL_DATE)["filename_components_hyphened"][1]

                    VAR = deconstruct_filename(DIR_VAR_VARSOTHER)["filename_components_hyphened"][0]
                    try:
                        OTHERVARS = deconstruct_filename(DIR_VAR_VARSOTHER)["filename_components_hyphened"][1]
                    except Exception as err:
                        print(DIR_SCOREVER, deconstruct_filename(DIR_VAR_VARSOTHER)["filename_components_hyphened"])
                        raise err

                    tmp = deconstruct_filename(LIST_DIR_SCOREVER[0])["filename_components_hyphened"][0]
                    if tmp=="rasters":
                        continue
                    SCOREVER = tmp
                    
                    if df_var_exists:
                        OUT_STRINGS.append(f"* {ANIMAL}|{DATE}| {VAR} | {OTHERVARS} |{SCOREVER}|{df_var_exists}")
                    else:
                        OUT_STRINGS.append(f"  {ANIMAL}|{DATE}| {VAR} | {OTHERVARS} |{SCOREVER}|{df_var_exists}")
     
    ts = makeTimeStamp()
    OUT_STRINGS = [f"Checked on {ts}"] + OUT_STRINGS
   
    path = f"{DEST_DIR}/anova_df_var_exists.txt"
    writeStringsToFile(path, OUT_STRINGS)
    print("Saved log results to: ", path)

def rec_session_durations_extract_kilosort(animal, date):
    """
    Find the durations of data for each session that has been completely preprocessed and also kilosorted.
    Purpose was for re-zeroing spike times from each session relative to the onset of that session, for ks.
    Finds durations by going through raw data logs.

    NOTE: Collects RS4 durations separately for each RS, since they can differ by few ms in their final samples, whichi
    matters if you are collecting mulktple sessions across day...

    # Confident about the following:
    # - kiloosrt spike times are correct relative to onset of each sessions neural data (RS4).
    # - total duration of file used in KS will be same duration as sum of raw RS4 durations.
    # - data tank (e.g., all behavior and events) are at most 0.2 sec offset, from neural data, and I am checking with
    # Myles whether this is guaranteed to be at the offset.
    # Sanity check, ks similar to tdt, e.g, find M1 channel and do crosscor.

    :param animal:
    :param date:
    :return:
    durations_each_sess_rs4, duration_total_kilosort, _durations_each_sess_using_tank
    First 2 are good.
    """
    from neuralmonkey.utils.directory import find_rec_session_paths, find_ks_cluster_paths
    from pythonlib.tools.exceptions import NotEnoughDataException
    from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED, PATH_DATA_NEURAL_RAW
    import pickle
    import scipy.io as sio
    import numpy as np

    # (1) Find list of sessions and ensur ethey are aligned between ks and nerual preprocess
    sessions_rec = find_rec_session_paths(animal, date)
    sessions_ks = find_ks_cluster_paths(animal, date)

    # sanity check that the sessions are aligned between rec and ks
    if not len(sessions_ks)==len(sessions_rec):
        print(len(sessions_ks), len(sessions_rec))
        print("sessions_ks:")
        for x in sessions_ks:
            print(x)
        print("sessions_rec:")
        for x in sessions_rec:
            print(x)
        print("you probably excluded some neural sessions for final analysis (moved to recordings_IGNORE). No solution yet for this problem.")
        assert False, "breaking, becuase you probably want to fix this problem (just re-do all of kilosort, prob ran kilosrt before you removed bad neural sessions, this is rare)."
        # raise NotEnoughDataException

    # - - chekck that the names match for neural and ks.
    for sessks, sessrec in zip(sessions_ks, sessions_rec):
        assert sessks["pathfinal"] == sessrec["pathfinal"]

    ################# DIFFERENT METHODS TO FIND DURATIONS OF EACH SESSION
    # (1) Use data tanks that are cached. THIS IS NOT PERFECTLY accurate, since tank times are slignly shorter than rs4.
    _durations_each_sess_using_tank = []
    for sessrec in sessions_rec:
        path = f"{PATH_DATA_NEURAL_PREPROCESSED}/recordings/{animal}/{date}/{sessrec['pathfinal']}/data_tank.pkl"
        with open(path, "rb") as f:
            dattank = pickle.load(f)

        duration_sec = dattank["info"]["duration"].total_seconds()
        _durations_each_sess_using_tank.append(duration_sec)

        # OTher alternative methods to get duration, but seem to be less than above
        if False:
            fs = dattank["streams"]["PhDi"]["fs"]
            nsamp = len(dattank["streams"]["PhDi"]["data"])
            nsamp/fs
            fs = dattank["streams"]["Mic1"]["fs"]
            nsamp = len(dattank["streams"]["Mic1"]["data"])
            nsamp/fs
            fs = dattank["streams"]["PhDi"]["fs"]
            nsamp = len(dattank["streams"]["PhDi"]["data"])

    # # (2) Duration of total of neural data used in kilosort, in the raw data.
    # # i.e. raw(RS4) --> concated across sessions --> saved [THIS DURATION] --> kilosort ...
    # duration_total_kilosort = None
    # list_batchnames = ["RSn2_batch1", "RSn2_batch2", "RSn3_batch1", "RSn3_batch2"]
    # duration_total_kilosort_dict = {}
    # for batchname in list_batchnames:
    #     dirpath = f"{PATH_KS_RAW}/{animal}/{date}/{batchname}" # choose any batch, they are identical.
    #     file = "ops.mat"
    #     path = f"{dirpath}/{file}"
    #     if os.path.exists(path):
    #         mat_dict = sio.loadmat(path)
    #         FS = mat_dict["ops"]["fs"][0][0][0][0]
    #         assert FS == 24414.0625, f"{FS}, why is this different?"
    #         sampsToRead = mat_dict["ops"]["sampsToRead"][0][0][0][0]
    #         tend = mat_dict["ops"]["tend"][0][0][0][0]
    #         assert tend==sampsToRead, "figure out which one is correct -- num samps"
    #         # Duration combining all batches
    #         if duration_total_kilosort is None:
    #             duration_total_kilosort = sampsToRead/FS
    #         else:
    #             # confirm not different
    #             assert duration_total_kilosort - sampsToRead/FS < 0.005
    #         # Collect durations for each rs and back of chans
    #         duration_total_kilosort_dict[batchname] = sampsToRead/FS

    # (2) Duration of total of neural data used in kilosort, in the raw data (data that was concatted during Kilosort pipeline)
    # i.e. raw(RS4) --> concated across sessions --> saved [THIS DURATION] --> kilosort ...
    duration_total_kilosort_dict_each_rs = {}
    for rsnum in [2,3]:
        _duration_this_rs = None
        for batchnum in [1,2,3,4]:
            dirpath = f"{PATH_KS_RAW}/{animal}/{date}/RSn{rsnum}_batch{batchnum}" # choose any batch, they are identical.
            file = "ops.mat"
            path = f"{dirpath}/{file}"
            if os.path.exists(path):
                mat_dict = sio.loadmat(path)
                FS = mat_dict["ops"]["fs"][0][0][0][0]
                assert FS == 24414.0625, f"{FS}, why is this different?"
                sampsToRead = mat_dict["ops"]["sampsToRead"][0][0][0][0]
                tend = mat_dict["ops"]["tend"][0][0][0][0]
                assert tend==sampsToRead, "figure out which one is correct -- num samps"
                # Duration combining all batches
                if _duration_this_rs is None:
                    _duration_this_rs = sampsToRead/FS
                else:
                    # confirm not different
                    assert _duration_this_rs - sampsToRead/FS < 0.001

        # Collect durations for each rs and back of chans
        assert _duration_this_rs is not None
        duration_total_kilosort_dict_each_rs[rsnum] = _duration_this_rs


    # (3) Duration of lenght of each RS4 recordings, saved in raw logs
    # i.e., raw(RS4) [THIS, in logs] --> concated..
    durations_each_sess_rs4_keyed_by_rs = {}
    rs_missed = []
    for rs in [2, 3]:
        try:
            # Collect durations across all sessions.
            durations = [] # list, length sessions/
            for sessnum, sessrec in enumerate(sessions_rec):
                # - Collect duration for this session
                logfile = f"RSn{rs}_log"
                path = f"{PATH_DATA_NEURAL_RAW}/{animal}/{date}/{sessrec['pathfinal']}/{logfile}.txt"
                with open(path) as f:
                    lines = f.readlines()

                if len(lines)>2:
                    # Then is something like this. Keep first and last.
                    # ['recording started at sample: 2\n', 'gap detected. last saved sample: 51833413, new saved sample: 51833425\n', 'recording stopped at sample: 332994022\n']
                    lines = [lines[0], lines[-1]]

                try:
                    assert lines[0][:27] == 'recording started at sample'
                    assert lines[1][:20] == 'recording stopped at'
                except AssertionError as err:
                    print("==========")
                    print(lines)
                    print(len(lines))
                    for l in lines:
                        print(l)
                    print(rs, sessnum, sessrec, path)
                    assert False, "investigate..."

                ind1 = lines[0].find(": ")
                ind2 = lines[0].find("\n")
                samp_on = int(lines[0][ind1+2:ind2])
                assert samp_on < 25, "why is RS4 signal offset from onset of trial. This probably means misalignment vs. Data tank..."

                ind1 = lines[1].find(": ")
                ind2 = lines[1].find("\n")
                samp_off = int(lines[1][ind1+2:ind2])
                nsamp = samp_off - samp_on + 1
                # if dur is None:
                #     dur = nsamp/FS
                # else:
                #     assert dur - nsamp/FS < 0.005

                durations.append(nsamp/FS)
        except FileNotFoundError as err:
            rs_missed.append(rs)
            durations = None

        durations_each_sess_rs4_keyed_by_rs[rs] = durations
        # # Store across all (sess, rs)
        # durations_each_sess_rs4_keyed_by_sessnum_rs_dict[(sessnum, rsnum)] = nsamp/FS

        # durations_each_sess_rs4.append(dur)
    if rs_missed == [2]:
        # Then use 3 to replace 2:
        durations_each_sess_rs4_keyed_by_rs[2] = durations_each_sess_rs4_keyed_by_rs[3]
    elif rs_missed == [3]:
        durations_each_sess_rs4_keyed_by_rs[3] = durations_each_sess_rs4_keyed_by_rs[2]
    elif rs_missed == [2,3]:
        assert False, "did not find RSn logs (e.g, FileNotFoundError: [Errno 2] No such file or directory: '/home/lucas/mnt/Freiwald/ltian/recordings/Diego/240523/Diego-240523-155459/RSn3_log.txt')"
    else:
        # Good
        assert rs_missed == []

    # (4) Total duration, by summing up RS4 raw across sessions (from log files).
    # sessnums = sorted(set([x[0] for x in out["durations_each_sess_rs4_keyed_by_sessnum_rs_dict"].keys()]))
    # sessnums = sorted(set([x[0] for x in durations_each_sess_rs4_keyed_by_sessnum_rs_dict.keys()]))
    duration_total_by_summing_rs4_dict = {}
    for rs in [2,3]:
        duration_total_by_summing_rs4_dict[rs] = sum(durations_each_sess_rs4_keyed_by_rs[rs])
        # duration_total_by_summing_rs4_dict[rs] = sum([durations_each_sess_rs4_keyed_by_sessnum_rs_dict[(s, rs)] for s in sessnums])

    ## Sanity check, durations for each sess add up to the total duration
    for rs in [2,3]:
        assert duration_total_by_summing_rs4_dict[rs] - duration_total_kilosort_dict_each_rs[rs]<0.001

    # Sanity check, dont expect tank to be accurate but m ake sure it is not totlaly worng relative to neural.
    for rs in [2,3]:
        durations = durations_each_sess_rs4_keyed_by_rs[rs]
        for dur1, dur2 in zip(_durations_each_sess_using_tank, durations):
            assert dur1-dur2 < 0.15, "Problem probably in getting durations from RSn2_log, in the string parsing part?"

    # Get onset time of session, using RS4 log data for each session.
    onsets_using_rs4_each_rs ={}
    offsets_using_rs4_each_rs ={}
    for rs in [2,3]:
        durations = durations_each_sess_rs4_keyed_by_rs[rs]

        onsets = [0.] + list(np.cumsum(durations)[:-1])
        offsets = [a+b for a,b in zip(onsets, durations)]

        onsets_using_rs4_each_rs[rs] = onsets
        offsets_using_rs4_each_rs[rs] = offsets

    out = {
        # "durations_each_sess_rs4":durations_each_sess_rs4,
        # "duration_total_kilosort":duration_total_kilosort,
        "_durations_each_sess_using_tank":_durations_each_sess_using_tank,
        "onsets_using_rs4_each_rs":onsets_using_rs4_each_rs,
        "offsets_using_rs4_each_rs":offsets_using_rs4_each_rs,
        # "durations_each_sess_rs4_keyed_by_sessnum_rs_dict":durations_each_sess_rs4_keyed_by_sessnum_rs_dict,
        "durations_each_sess_rs4_keyed_by_rs":durations_each_sess_rs4_keyed_by_rs,
        # "duration_total_kilosort_dict":duration_total_kilosort_dict,
        "duration_total_kilosort_dict_each_rs":duration_total_kilosort_dict_each_rs,
        "duration_total_by_summing_rs4_dict":duration_total_by_summing_rs4_dict,
    }

    print("These durations gotten for sessions...")
    for k, v in out.items():
        print("... ", k, ":", v)
    # print("... durations_each_sess_rs4_keyed_by_rs", durations_each_sess_rs4_keyed_by_rs)
    # print("... _durations_each_sess_using_tank", _durations_each_sess_using_tank)
    # print("... duration_total_by_summing_rs4_dict", duration_total_by_summing_rs4_dict)

    return out


if __name__=="__main__":
    import sys
    animal = sys.argv[1]
    analy = sys.argv[2]

    if analy=="preprocess":
        check_log_preprocess_completion_status(animal)
    elif analy=="anova":
        check_log_anova_analy_status()
    else:
        print(animal)
        print(analy)
        assert False

