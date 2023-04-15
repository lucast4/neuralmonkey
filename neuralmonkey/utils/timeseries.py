import numpy as np
import matplotlib.pyplot as plt

def dat_to_time(vals, fs):
    """ Given sample data, get time bins for vals
    PARAMS:
    - vals is len N array
    - fs, sample rate in Hz
    RETURNS:
    - t, len N array of time, assuming first bin is time 0
    """
    t = np.arange(len(vals))
    t = t/fs
    return t

def convert_discrete_events_to_time_series(t0, tend, event_onsets, event_offsets, fs,
    ploton=False, clamp_onset_to_this=None):
    """ Given times of discrete events, return time series where 1 means even is occuring
    PARAMS:
    - t0, time of first bin, inclusive
    - tend, time of last bin, inclusvine
    - event_onsets, array of times of onests, inclusvive
    - event_offsets, array of times of offsets (same len as event_onsets), inclusvive
    - fs, sampling rate (for output resolution)
    RETURNS:
    - times, 
    - vals
    """

    # allow for diff in length of 1, will pad either onsets or offsets if thats the case
    if len(event_onsets)<len(event_offsets):
        # assume onset is at time 0
        event_onsets = np.insert(event_onsets, 0, t0)
    elif len(event_onsets)>len(event_offsets):
        # then append to end of offsets
        event_offsets = np.append(event_offsets, tend)

    assert len(event_onsets)==len(event_offsets)

    # generate time bins and vals
    # fs = 1000.
    period = 1/fs
    times = np.arange(t0, tend+period, period) # inclusive.
    vals = np.zeros_like(times)

    def find_nearest_time_bin_(time):
        """ Get time bin that is closests to time
        """
        return np.argmin(np.abs(times - time))

    for on, off in zip(event_onsets, event_offsets):
        assert off > on

        indon = find_nearest_time_bin_(on)
        indoff = find_nearest_time_bin_(off)

        vals[indon:indoff+1] = 1.

    # clamp_onset_to_this = 1.
    # vals = 1-vals

    if ploton:
        fig, ax = plt.subplots()
        ax.plot(times, vals, label="before clamping onset")

    # help clean it up by forcinfg the onset to be a certain bvalue. 
    # e.g, onset of this trial cannot be positive, if so, then is remnant of previous trial.
    # remove it.
    if clamp_onset_to_this==0.:
        if vals[0]==1.:
            inds = np.where(np.diff(vals)==-1.)[0]
            vals[:inds[0]+1] = 0.
            if inds[0]/len(vals)>0.1:
                ax.plot(times, vals, '--r', label="after clamping onset")
                ax.legend()                
                print(inds)
                print(len(vals))
                assert False, "this may be bug. expect clamping to not extend this far into the time series."
    elif clamp_onset_to_this==1.:
        if vals[0]==0.:
            inds = np.where(np.diff(vals)==1.)[0]
            vals[:inds[0]+1] = 1.
            if inds[0]/len(vals)>0.1:
                ax.plot(times, vals, '--r', label="after clamping onset")
                ax.legend()                
                print(inds)
                print(len(vals))
                assert False, "this may be bug. expect clamping to not extend this far into the time series."
    else:
        assert clamp_onset_to_this is None

    if ploton:
        ax.plot(times, vals, '--r', label="after clamping onset")
        ax.legend()

    return times, vals

