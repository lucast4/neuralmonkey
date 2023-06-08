import numpy as np

def dfthis_to_frmat(dfthis, times=None, fr_ver="fr_sm", time_bin_size=None,
        output_dim=2):
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

    if time_bin_size:
        if times is None:
            times = dfthis.iloc[0]["fr_sm_times"].squeeze()

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
    if output_dim==3:
        frmat = frmat[:, None, :]
    else:
        assert output_dim==2

    return frmat, times