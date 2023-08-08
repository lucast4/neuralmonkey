""" Working with pathsa nd directories
"""

from pythonlib.globals import PATH_NEURALMONKEY, PATH_DATA_NEURAL_RAW, PATH_DATA_NEURAL_PREPROCESSED
from pythonlib.tools.expttools import writeStringsToFile, makeTimeStamp

def find_rec_session_paths(animal, date):
    """
    """
    from pythonlib.tools.expttools import findPath, deconstruct_filename

    path_hierarchy = [
        [animal],
        [date]
    ]
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

    # go thru raw data to collect dates and sessions
    DIR_RAW = f"{PATH_DATA_NEURAL_RAW}/{animal}"
    list_dir_date = glob.glob(f"{DIR_RAW}/*")

    res = []
    res_strings = []
    for dirthis in list_dir_date:

        # List of sessiosn for this date
        list_dir_sessions = glob.glob(f"{dirthis}/*")
        # print(dirthis)
        date = int(fileparts(dirthis)[-2])
        
        # count and check sessions
        date_string = f"{date}" # Initialize
        for i, dirsess in enumerate(list_dir_sessions):
            
            sess_id = fileparts(dirsess)[-2] # e.g, Pancho-220715-154205
            DIR_PREPROCESS = f"{PATH_DATA_NEURAL_PREPROCESSED}/recordings/{animal}/{date}/{sess_id}"
            DIR_PREPROCESS_CACHED = f"{DIR_PREPROCESS}/cached"

            # check if preprocess done.
            SN = load_session_helper(date, None, i, animal, None,  
                ACTUALLY_BAREBONES_LOADING = True) # Very quick loading
            SN.Paths = {
                "cached_dir":DIR_PREPROCESS_CACHED
            }
            done = SN._savelocalcached_check_done(datslice_quick_check=True)
            
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
            
        res_strings.append({
            "date":date,
            "string":date_string
        })

                
    # write to file, after sorting by date
    res_strings = sorted(res_strings, key=lambda x: x["date"])
    list_strings = [x["string"] for x in res_strings]
    ts = makeTimeStamp()
    list_strings = [f"Checked on {ts}"] + ["date | sess-preprocess_done", "------------------------"] + list_strings
    writeStringsToFile(DEST_FILE, list_strings)

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

