def dat_to_time(vals, fs):
    """ Given sample date, get time bins for vals, where
    vals is len N array, assuming first bin is time 0
    """
    import numpy as np
    t = np.arange(len(vals))
    t = t/fs
    return t

