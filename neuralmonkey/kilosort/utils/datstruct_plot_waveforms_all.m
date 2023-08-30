function datstruct_plot_waveforms_all(DATSTRUCT, SAVEDIR_FINAL, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR, plot_decision_boundaries)

if ~exist('plot_decision_boundaries', 'var'); plot_decision_boundaries = true; end

% Make final plots of all waveforms

%% Plot waveforms at decision boundaries
if plot_decision_boundaries
    close all;
    
    savedir = [SAVEDIR_FINAL '/waveform_gridplots'];
    mkdir(savedir);
    
    
    % 1) snr-noise_mua_boundary
    path_noext = [savedir '/snr-noise_mu_boundary'];
    values = [DATSTRUCT.snr_final];
    plot_waveforms_sorted_by(DATSTRUCT, values, [MIN_SNR-0.5, MIN_SNR+0.6], path_noext);
    
    % 1) snr-noise_mua_boundary
    path_noext = [savedir '/snr-mu_su_boundary'];
    values = [DATSTRUCT.snr_final];
    values_sort =  [THRESH_SU_SNR-1, THRESH_SU_SNR+1000];
    exclude_labels = {'noise'};
    plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
        path_noext, exclude_labels);
    
    path_noext = [savedir '/isi-mu_su_boundary'];
    values = [DATSTRUCT.isi_violation_pct];
    exclude_labels = {'noise'};
    plot_waveforms_sorted_by(DATSTRUCT, values, [THRESH_SU_ISI-1, THRESH_SU_ISI+0.03], ...
        path_noext, exclude_labels);
    
    % 1) snr-noise_mua_boundary
    path_noext = [savedir '/sharpiness-artifact_boundary'];
    values = [DATSTRUCT.sharpiness];
    plot_waveforms_sorted_by(DATSTRUCT, values, [THRESH_ARTIFACT_SHARP-10, THRESH_ARTIFACT_SHARP+1000], path_noext);
    
    % 1) snr-noise_mua_boundary
    path_noext = [savedir '/isi-artifact_boundary'];
    values = [DATSTRUCT.isi_violation_pct];
    plot_waveforms_sorted_by(DATSTRUCT, values, [THRESH_ARTIFACT_ISI-0.05, THRESH_ARTIFACT_ISI+1], path_noext);
    
end

%% Plot all examples of each label

savedir = [SAVEDIR_FINAL '/waveform_gridplots_byfinallabel'];
mkdir(savedir);

path_noext = [savedir '/ALL-sorted_by_changlobal'];
values = [DATSTRUCT.chan_global];
exclude_labels = {'noise', 'artifact'};
plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
    path_noext, exclude_labels);

path_noext = [savedir '/SU-sorted_by_snr'];
values = [DATSTRUCT.snr_final];
exclude_labels = {'noise', 'mua', 'artifact'};
plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
    path_noext, exclude_labels);

path_noext = [savedir '/SU-sorted_by_isi'];
values = [DATSTRUCT.isi_violation_pct];
exclude_labels = {'noise', 'mua', 'artifact'};
plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
    path_noext, exclude_labels);

path_noext = [savedir '/SU-sorted_by_changlobal'];
values = [DATSTRUCT.chan_global];
exclude_labels = {'noise', 'mua', 'artifact'};
plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
    path_noext, exclude_labels);

% path_noext = [savedir '/MU_SU-sorted_by_changlobal'];
% values = [DATSTRUCT.chan_global];
% exclude_labels = {'noise', 'artifact'};
% plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
%     path_noext, exclude_labels);

path_noext = [savedir '/artifact-sorted_by_sharpiness'];
values = [DATSTRUCT.sharpiness];
exclude_labels = {'noise', 'mua', 'su'};
plot_waveforms_sorted_by(DATSTRUCT, values, [], ...
    path_noext, exclude_labels);

%% SANITY, all should be GOOD.

% Plot all that are good labels, yet not labeled good.
inds1 = ismember({DATSTRUCT.label_final}, {'mua', 'su'}) & [DATSTRUCT.GOOD]==0;
inds2 = ismember({DATSTRUCT.label_final}, {'mua', 'su'}) & [DATSTRUCT.GOOD]==1;

disp(['Of mu and su, how many not called GOOD: ' num2str(sum(inds1)) '/' num2str(sum(inds2))]);

path_noext = [savedir '/SU_MU-GOOD_failed_'];
values = [DATSTRUCT.GOOD];
exclude_labels = {'noise', 'artifact'};
plot_waveforms_sorted_by(DATSTRUCT, values, [-0.1 0.1], ...
    path_noext, exclude_labels);

end