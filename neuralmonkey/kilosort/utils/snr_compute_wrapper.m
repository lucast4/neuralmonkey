function [snr_final, snr_new, peak_to_trough, waveforms_running_std, snr_old, peak_to_trough_old, waveforms_running_std_old, ...
        waveforms_aligned] = ...
        snr_compute_wrapper(waveforms, indpeak, npre, npost)
    % Autoamti ally doees stuff to waveforms to condition it for computing snr. Just
    % pass in the raw waveforms.

    [~, ~, ~, ~, ~, ~, ...
        ~, ~, ~, ...
        ~, SNR_VER] = quickscript_config_load()
    
    % remove any wf that are nan
    indsgood = ~isnan(waveforms(:,1));
    waveforms = waveforms(indsgood, :);

    switch SNR_VER
        case 'median'
            % no ned to remove outlsier
        case 'mean'
            % rmeov outlseir
            [waveforms, ~] = waveforms_remove_outliers(waveforms);
        otherwise
            assert(false);
    end 

    ispos = is_positive_waveform(waveforms);
    [waveforms_aligned, indpeak_new] = get_shifted_wf(waveforms, ispos, indpeak, ...
        npre, npost);

    [snr_old, peak_to_trough_old, waveforms_running_std_old] = snr_compute(waveforms, ...
         indpeak, npre, npost, SNR_VER);
    [snr_new, peak_to_trough, waveforms_running_std] = snr_compute(waveforms_aligned, ...
        indpeak_new, npre, npost, SNR_VER);

    snr_final = max([snr_old, snr_new]);
end