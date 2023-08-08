# from pythonlib.globals import PATH_DATA_BEHAVIOR_RAW, PATH_DATA_BEHAVIOR_RAW_SERVER, PATH_MATLAB
from pythonlib.globals import PATH_MATLAB, MACHINE

def spikes_extract_quick_tdt(animal, date, machine=None):
    import os
    # print("MATLAB: Converting bhv2 to h5")
    # print(f"matlab -nodisplay -nosplash -nodesktop -r \"convert_format('h5', '{f}'); quit\"")
    if machine is None:
        # use the local
        if MACHINE == "lucast4-MS-7B98":
            machine = "gorilla"
        else:
            machine = MACHINE

    print("RUNNING TDT EXTRACTION, this command to matlab:")
    command = f"{PATH_MATLAB} -nodisplay -nosplash -nodesktop -r \"extract_neural_quick_function('{date}', '{animal}', '{machine}', false); quit\""
    os.system(command)

