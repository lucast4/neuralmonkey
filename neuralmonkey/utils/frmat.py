import numpy as np

def bin_frmat_in_time(frmat, times, time_bin_size, slide=None):
    """
    Bin firing rates by time.
    frmat, last dim is time. Works for both 3-dim and 2-dim frmat.
    - frmat, 2d array of fr, usually (chans, times)=
    - times, array of times
    - time_bin_size, in sec, window size
    - slide, time in sec to slide the window (cneter to center).
    EXAMPLE: if bin is 0.1 and slide 0.05, then
    returned times will be 0.05, 0.15, ..
    """
    from neuralmonkey.classes.population import PopAnal

    if isinstance(times, np.ndarray) and len(times.shape)>0:
        times = times.squeeze()

    if slide is None:
        slide = time_bin_size
    else:
        if slide>time_bin_size:
            print(time_bin_size, slide)
            assert False


    # try:
    #     len(times)
    #     assert len(frmat.shape)==2
    #     assert len(frmat)>0
    #     assert frmat.shape[1]==len(times)
    # except Exception as err:
    #     print(times)
    #     print(type(times))
    #     raise err

    # Convert frmat to (:, :, times)
    INPUT_NDIMS = len(frmat.shape)
    if len(frmat.shape)==2:
        X = frmat[:frmat.shape[0], None, :frmat.shape[1]] # (nchans, 1, times)
        assert X.shape[0]==frmat.shape[0]
        assert X.shape[2]==frmat.shape[1]
    elif len(frmat.shape)==3:
        X = frmat
    else:
        print(frmat.shape)
        assert False

    pa = PopAnal(X, times)
    try:
        pa = pa.agg_by_time_windows_binned(time_bin_size, slide)
    except Exception as err:
        print("--------------")
        print(X.shape)
        print(times)
        print(time_bin_size)
        print(slide)
        raise err

    if INPUT_NDIMS==2:
        frmat = pa.X[:, 0, :]
    elif INPUT_NDIMS==3:
        frmat = pa.X
    else:
        assert False

    times = pa.Times

    return frmat, times

    # if time_axis is None:
    #     time_axis = len(frmat.shape)-1
    #
    # if isinstance(times, np.ndarray) and len(times.shape)>0:
    #     times = times.squeeze()
    #
    # MINDUR = time_bin_size/4;
    # # MINDUR = 0.05
    #
    # binedges = np.arange(times[0], times[-1]+time_bin_size, time_bin_size)
    # # add small value to avoid numeriacl imprecision errors.
    # binedges = binedges-0.001*time_bin_size
    # # print(times)
    # # print(binedges)
    # inds_bin = np.digitize(times, binedges)
    # inds_bin_unique = np.sort(np.unique(inds_bin))
    # # print("--", times[0], times[-1], time_bin_size)
    # # print(inds_bin)
    # # assert False
    #
    # list_t =[]
    # list_frvec = []
    # for binid in inds_bin_unique:
    #     indsthis = inds_bin==binid
    #
    #     if sum(indsthis)==0:
    #         continue
    #
    #     timesthis = times[indsthis]
    #     dur = max(timesthis) - min(timesthis)
    #     if dur<MINDUR:
    #         # print("Skipping bin: ", binid, dur)
    #         continue
    #
    #     # print(frmat.shape)
    #     # print(indsthis)
    #     # print(len(indsthis))
    #     frmatthis = frmat[..., indsthis]
    #
    #     t = np.mean(timesthis)
    #     # print(timesthis)
    #     # print(inds_bin, binid)
    #     # assert False
    #     frvec = np.mean(frmatthis, axis=time_axis)
    #     # print(frvec.shape)
    #     # assert False
    #
    #     list_t.append(t)
    #     list_frvec.append(frvec)
    #
    #
    # # concat
    # times = np.stack(list_t) # (nbins, )
    # # print(list_frvec[0].shape)
    # # assert False
    # frmat = np.stack(list_frvec, axis=time_axis) # (ndat, nbins)
    # # print("--", frmat.shape)

    return frmat, times




def dfthis_to_frmat(dfthis, times=None, fr_ver="fr_sm", time_bin_size=None,
        output_dim=2, return_as_zscore=False):
    """ convert dataframe, column name fr_ver, each itme is array
    of fr across time (1, ntimes), to frmat
    PARAMS:
    - time_bin_size, with option to further bin. sec, then bins starting from left time edge
    - times, list or array (ntimes,). only needed if time_bin_size. If None, then looks for it in df["fr_sm_times"]
    - output_dim, int, either {2, 3}. if 2, then frmat shape is (ndat, ntimes), else if
    (ndat, 1, ntimes).
    RETURNS:
    - frmat, shape, see abov
    - times
    """

    frmat = np.concatenate(dfthis[fr_ver].tolist(), axis=0)    
    if times is None:
        times = dfthis.iloc[0]["fr_sm_times"].squeeze()

    if time_bin_size:

        MINDUR = time_bin_size/4;
        # MINDUR = 0.05

        binedges = np.arange(times[0], times[-1]+time_bin_size, time_bin_size)
        # add small value to avoid numeriacl imprecision errors.
        binedges = binedges-0.001*time_bin_size
        # print(times)
        # print(binedges)
        inds_bin = np.digitize(times, binedges)
        inds_bin_unique = np.sort(np.unique(inds_bin)) 
        # print("--", times[0], times[-1], time_bin_size)
        # print(inds_bin)
        # assert False

        list_t =[]
        list_frvec = []
        for binid in inds_bin_unique:
            indsthis = inds_bin==binid
            
            if sum(indsthis)==0:
                continue
                
            timesthis = times[indsthis]
            dur = max(timesthis) - min(timesthis)
            if dur<MINDUR:
                # print("Skipping bin: ", binid, dur)
                continue
            frmatthis = frmat[:, indsthis]
            
            t = np.mean(timesthis)
            frvec = np.mean(frmatthis, axis=1)
            
            list_t.append(t)
            list_frvec.append(frvec.T)
            
            
        # concat
        times = np.stack(list_t) # (nbins, )s
        frmat = np.stack(list_frvec, axis=1) # (ndat, nbins)
        # print("--", frmat.shape)

    if return_as_zscore:
        def _frmat_convert_zscore(frmat):
            """ convert to a single zscore trace, using the grand mean and std.
            Returns same shape as input
            """
            m = np.mean(frmat[:], keepdims=True)
            s = np.std(frmat[:], keepdims=True)

            return (frmat - m)/s
        shin = frmat.shape
        frmat = _frmat_convert_zscore(frmat)
        shout = frmat.shape
        if not shin==shout:
            print(shin)
            print(shout)
            assert False

    if output_dim==3:
        frmat = frmat[:, None, :]
    else:
        assert output_dim==2

    return frmat, times