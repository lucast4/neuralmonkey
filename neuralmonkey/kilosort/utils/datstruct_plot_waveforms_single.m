function datstruct_plot_waveforms_single(DATSTRUCT, idx, NPLOT)
    % plots the unaligned or aligned wf, depending on which one
    % has highest signal to nose.
    
    % WHich waveforsm to plot?
    if ~isfield(DATSTRUCT, 'use_aligned_wf')
        % then use not-aligned
        wf = DATSTRUCT(idx).waveforms;
    elseif DATSTRUCT(idx).use_aligned_wf
        % Use algined
        wf = DATSTRUCT(idx).waveforms_aligned;
    else
        % then use not-aligned
        wf = DATSTRUCT(idx).waveforms;
    end
    plot_waveforms_singleplot(wf, NPLOT);
end
