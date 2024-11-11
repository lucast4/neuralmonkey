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


def timewarp_piecewise_linear_interpolate(X_trial, times_trial, anchor_times_this_trial, times_template, 
                                          anchor_times_template, smooth_boundaries_sigma=None,
                                          PLOT=False):
    """
    Piecewise linear time-warping of firing rates to a common time base using given anchor points.
    E.g, the anchor points are behavioral events, and you want to align this trial to a common (median) 
    set of events.

    All data in X_trial will be retained, just shifted in time. 

    The endpoints of times_trial and times_template are always anchored to each other

    IMPORTANT - the times in times_trial and times_template are invariant up to a shift. This is becuase they are
    forced to align by their endpoints. 

    Parameters:
    X_trial : np.ndarray
        Neural firing rates of shape (n_neurons, n_times) for a single trial (usually single trial)
    times_trial : np.ndarray
        Time points for the trial of shape (n_times,).
    anchor_times_this_trial : np.ndarray
        Anchor points in times_trial that correspond to events inside times_trials, shape (n_anchor_pts,). Should NOT 
        include the endpoints of times_trial
    times_template : np.ndarray
        Target time points for the template of shape (n_times_template,).
    anchor_times_template : np.ndarray
        Anchor points in times_template that correspond to events, shape (n_anchor_pts,).
    smooth_boundaries_sigma: None or scalar (in seconds, sigma for gaussian window) to smooth along time axis, the final output.
    Returns:
    X_warped : np.ndarray
        Warped neural firing rates of shape (n_neurons, n_times_template).

    CODE TO TEST:
        from neuralmonkey.utils.frmat import timewarp_piecewise_linear_interpolate

        # Example inputs
        X_trial = np.random.rand(10, 100)  # 10 neurons, 100 time points in trial
        X_trial =np.stack([np.linspace(0, 10, 100) for _ in range(5)], axis=1).T
        times_trial = np.linspace(0, 1, 100)  # Original trial time points


        anchor_times_this_trial = [0.5]  # Anchors in trial time
        times_template = np.linspace(0, 1, 120)  # Template with 120 time points
        anchor_times_template = [0.1]  # Anchors in template time

        delta = 0
        times_template = times_template+delta
        anchor_times_template = [t+delta for t in anchor_times_template]

        # Perform time-warping
        X_warped = timewarp_piecewise_linear_interpolate(
            X_trial, times_trial, anchor_times_this_trial, times_template, anchor_times_template,
            smooth_boundaries_sigma=smooth_sigma, PLOT=True
        )

    """
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    # Plotting code
    def plot_timewarp_results(X_trial, X_warped, times_trial, times_template):
        """
        Plots the original and time-warped firing rates for comparison.
        
        Parameters:
        X_trial : np.ndarray
            Original neural firing rates (n_neurons, n_times).
        X_warped : np.ndarray
            Warped neural firing rates (n_neurons, n_times_template).
        times_trial : np.ndarray
            Time points for the trial (n_times,).
        times_template : np.ndarray
            Time points for the template (n_times_template,).
        """
        fig, axes = plt.subplots(1,2, figsize=(12, 6))
        
        # Plot original trial data
        ax = axes.flatten()[0]
        img = ax.imshow(X_trial, aspect='auto', extent=[times_trial[0], times_trial[-1], 0, X_trial.shape[0]])
        plt.colorbar(img, label='Firing Rate')
        ax.set_xlabel('Original Trial Time')
        ax.set_ylabel('Neuron')
        ax.set_title('Original Trial Firing Rates')
        for t in anchor_times_this_trial:
             ax.axvline(t, color="r")
             
        # Plot time-warped data
        ax = axes.flatten()[1]
        img = ax.imshow(X_warped, aspect='auto', extent=[times_template[0], times_template[-1], 0, X_warped.shape[0]])
        plt.colorbar(img, label='Firing Rate')
        ax.set_xlabel('Template Time')
        ax.set_ylabel('Neuron')
        ax.set_title('Time-Warped Firing Rates to Template')
        for t in anchor_times_template:
             ax.axvline(t, color="r")

        plt.tight_layout()
        return fig


    # Ensure the number of anchors match
    if len(anchor_times_this_trial) != len(anchor_times_template):
        raise ValueError("Anchor points in trial and template must have the same length.")
    
    # anchor times should only be inner. 
    assert anchor_times_this_trial[0]>times_trial[0] and anchor_times_this_trial[-1]<times_trial[-1]
    assert anchor_times_template[0]>times_template[0] and anchor_times_template[-1]<times_template[-1]

    # append the endpoints as anchors always
    anchor_times_this_trial = np.append(np.insert(anchor_times_this_trial, 0, times_trial[0]), times_trial[-1])
    anchor_times_template = np.append(np.insert(anchor_times_template, 0, times_template[0]), times_template[-1])

    # Anchor times must be monotically increasing
    assert np.all(np.diff(anchor_times_this_trial)>0)
    assert np.all(np.diff(anchor_times_template)>0)

    # Initialize an array to store the warped firing rates
    n_neurons = X_trial.shape[0]
    X_warped = np.zeros((n_neurons, len(times_template))) - np.inf
    
    # Loop through each segment defined by the anchor points
    for i in range(len(anchor_times_this_trial) - 1):
        # Get start and end points for each segment in trial and template
        start_trial, end_trial = anchor_times_this_trial[i], anchor_times_this_trial[i + 1]
        start_template, end_template = anchor_times_template[i], anchor_times_template[i + 1]
        
        # Mask the times in the current segment for the trial
        mask_trial = (times_trial >= start_trial) & (times_trial <= end_trial)
        if ~np.any(mask_trial):
            print(anchor_times_this_trial)
            assert False, "anchor times are too close"
        time_segment_trial = times_trial[mask_trial]
        firing_segment = X_trial[:, mask_trial]
        
        # Mask for the template time segment
        mask_template = (times_template >= start_template) & (times_template <= end_template)
        time_segment_template = times_template[mask_template]
        if ~np.any(mask_template):
            print(times_template)
            assert False, "anchor times are too close"
        
        # Normalize time segments to [0, 1]
        normalized_trial = (time_segment_trial - start_trial) / (end_trial - start_trial)
        normalized_template = (time_segment_template - start_template) / (end_template - start_template)
        
        # Interpolate firing rates to match the normalized template time segment
        interp_func = interp1d(normalized_trial, firing_segment, axis=1, fill_value="extrapolate")
        warped_segment = interp_func(normalized_template)

        # Insert the remaining warped segment into the final output array
        X_warped[:, mask_template] = warped_segment

        # # Insert the warped segment into the final output array
        # X_warped[:, mask_template] = warped_segment

    assert ~np.any(np.isinf(X_warped)), "failed to fill some values..."

    # Apply Gaussian smoothing across the time axis to smooth out segment boundaries
    if smooth_boundaries_sigma is not None:
        from scipy.ndimage import gaussian_filter1d
        period = np.mean(np.diff(times_template))
        smooth_sigma_bins = np.ceil(smooth_boundaries_sigma/period)
        X_warped = gaussian_filter1d(X_warped, sigma=smooth_sigma_bins, axis=1)

    if PLOT:
        # Plot the results
        fig1, ax = plt.subplots()
        ax.plot(times_trial, X_trial[0,:], label="orig")
        ax.plot(times_template, X_warped[0,:], label="warped")
        ax.legend()
        fig2 = plot_timewarp_results(X_trial, X_warped, times_trial, times_template)    
    else:
        fig1, fig2 = None, None

    return X_warped, fig1, fig2

