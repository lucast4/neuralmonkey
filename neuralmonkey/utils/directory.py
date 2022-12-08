""" Working with pathsa nd directories
"""

from pythonlib.globals import PATH_NEURALMONKEY, PATH_DATA_NEURAL_RAW, PATH_DATA_NEURAL_PREPROCESSED

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


