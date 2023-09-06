function datstruct_plot_waveforms_single(DATSTRUCT, idx, NPLOT)
% plots the unaligned or aligned wf, depending on which one
% has highest signal to nose.

% WHich waveforsm to plot?

if ~exist('NPLOT', 'var'); NPLOT = 100; end

if ~isfield(DATSTRUCT, 'snr_aligned')
    % then use not-aligned
    wf = DATSTRUCT(idx).waveforms;
else
    snr_improvement = DATSTRUCT(idx).snr_aligned / DATSTRUCT(idx).snr_not_aligned;
    if snr_improvement>1.25
        % then use aligned
        wf = DATSTRUCT(idx).waveforms;
        ispos = is_positive_waveform(wf);
        [indpeak, wind_spike, npre, npost] = quickscript_config_load();
        [wf, indpeak_new] = get_shifted_wf(wf, ispos, indpeak, ...
            npre, npost);
    else
        wf = DATSTRUCT(idx).waveforms;
    end
end
plot_waveforms_singleplot(wf, NPLOT);
end
