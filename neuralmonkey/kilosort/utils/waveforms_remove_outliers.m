function [wf_outliersgone] = waveforms_remove_outliers(...
    wf, PLOT)
% Return waveforms with outliers removed.

if ~exist('PLOT', 'var'); PLOT = false; end

% wf = DATSTRUCT(idx).waveforms;

% DATSTRUCT(idx).snr_aligned = 124;
% DATSTRUCT(idx).snr_not_aligned = 100;
% DATSTRUCT(idx).waveforms_aligned = [];

% Keep trying until you dont remove any mre. This useful becuase first remove the
% really bad outliers, and then you can better estimate the IQR
wf_outliersgone = wf;
for i=1:5
    [wf_outliersgone, inds_remove] = waveforms_remove_outliers_inner(...
        wf_outliersgone, PLOT);
    if sum(inds_remove)==0
        break
    end
end

if PLOT
    assert(false, 'stop, or else too many plots...');
end

end