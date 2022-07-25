def dat_to_time(vals, fs):
    """ Given sample data, get time bins for vals
    PARAMS:
    - vals is len N array
    - fs, sample rate in Hz
    RETURNS:
    - t, len N array of time, assuming first bin is time 0
    """
    import numpy as np
    t = np.arange(len(vals))
    t = t/fs
    return t

