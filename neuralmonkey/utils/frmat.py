import numpy as np

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