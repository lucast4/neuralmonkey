function [snr, peak_to_trough, waveforms_running_std] = snr_compute(waveforms, ...
    indpeak, npre, npost, VER, PLOT)
    % waveforms (ntrials, ntimebins)
    % Given waveforms, compute snr

    % npre = 9;
    % npost = 16;

    if ~exist('PLOT', 'var'); PLOT = false; end

    waveforms_this = waveforms(:, indpeak-npre:indpeak+npost);

    switch VER
        case 'median'
            waveforms_mean = median(waveforms_this, 1);
        case 'mean'
            waveforms_mean = mean(waveforms_this, 1);
        otherwise
            assert(false);
    end
    peak_to_trough = max(waveforms_mean) - min(waveforms_mean);

    switch VER
        case 'median'
            tmp = prctile(waveforms_this, [50-34 50+34], 1);
            waveforms_running_std = tmp(2,:) - tmp(1,:);
        case 'mean'
            waveforms_running_std = std(waveforms_this, 0, 1);
        otherwise
            assert(false);
    end
    % take mean running std around time of spike
    std_window_mean = mean(waveforms_running_std);

    % also use the mean of 95th and 5th percentiles, to catch cases where the spike
    % is thin.
    tmp = prctile(waveforms_running_std, [2.5 97.5]);
    
    if PLOT
        disp(['std 5tha nd 95th prctiles (and mean): ' num2str([tmp mean(tmp)])]);
        disp(std_window_mean);
        
        figure; hold on;
        subplot(2,2,1); hold on;
        plot(waveforms');
        subplot(2,2,2); hold on;
        plot(waveforms_running_std);
        line(xlim, [peak_to_trough peak_to_trough]);
        line(xlim, [0 0]);
        line(xlim, [tmp(1) tmp(1)]);
        line(xlim, [std_window_mean std_window_mean]);
        line(xlim, [tmp(2) tmp(2)]);
        % assert(false);
    end

    std_window_mean = mean([std_window_mean mean(tmp)]);

    % FInal snr
    snr = peak_to_trough/std_window_mean;
end


