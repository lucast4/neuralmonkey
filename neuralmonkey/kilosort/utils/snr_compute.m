function [snr, peak_to_trough, waveforms_running_std] = snr_compute(waveforms, indpeak, npre, npost)
    % waveforms (ntrials, ntimebins)
    % Given waveforms, compute snr

    % npre = 9;
    % npost = 16;

    waveforms_this = waveforms(:, indpeak-npre:indpeak+npost);

    if true
        waveforms_mean = median(waveforms_this, 1);
    else
        waveforms_mean = mean(waveforms_this, 1);
    end
    peak_to_trough = max(waveforms_mean) - min(waveforms_mean);

    if true
        tmp = prctile(waveforms_this, [50-34 50+34], 1);
        waveforms_running_std = tmp(2,:) - tmp(1,:);
    else
        waveforms_running_std = std(waveforms_this, 0, 1);
    end
    % take mean running std around time of spike
    std_window_mean = mean(waveforms_running_std);

    snr = peak_to_trough/std_window_mean;
end


