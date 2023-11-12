function [wf_outliersgone, inds_remove] = waveforms_remove_outliers_inner(wf, PLOT)
% Inner function, called by waveforms_remove_outliers

%% outliers
% wf_orig = wf;

% Do 2 iterations, first remove really bad outliers if they exist.
vals_min = min(wf, [], 2);
vals_max = max(wf, [], 2);
vals_ss = sum(wf.^2, 2); % sum square

% Find indices with any failures (tukey-like outliers).
[~, inds_remove1] = remove_outliers(vals_min, 3, PLOT, 'low');
[~, inds_remove2] = remove_outliers(vals_max, 3, PLOT, 'high');
% [~, inds_remove3] = remove_outliers(vals_ss, 2.1, PLOT);
[~, inds_remove3] = remove_outliers(vals_ss, 2, PLOT, 'high');

% fail if you fail any criteria.
inds_remove = inds_remove1 | inds_remove2 | inds_remove3;

%% return the waveforms without poutliers.
wf_outliersgone = wf;
wf_outliersgone(inds_remove, :) = [];

if PLOT
    figure; hold on;
    subplot(2,2,1); hold on;
    title('all');
    % datstruct_plot_waveforms_single(DATSTRUCT, idx, 100);
    plot_waveforms_singleplot(wf, size(wf,1), false);
    % plot_waveforms_singleplot(wf(inds_remove, :), length(inds_remove));
    subplot(2,2,2); hold on;
    title('outleirs');
    plot_waveforms_singleplot(wf(inds_remove, :), length(inds_remove), false);
    subplot(2,2,3); hold on;
    title('not outleirs');
    plot_waveforms_singleplot(wf_outliersgone, length(~inds_remove), false);

    figure; hold on;
    subplot(2,2,1); hold on;
    title('mins and maxes (outliers marked)');
    plot(vals_min, 0, 'xk');
    plot(vals_max, 0, 'xr');
    if sum(inds_remove)>0
        plot(vals_min(inds_remove), 0, 'ok');
        plot(vals_max(inds_remove), 0, 'or');
    end
    
    subplot(2,2,2); hold on;
    title('sum square');
    plot(vals_ss, 0, 'xb');
    if sum(inds_remove)>0
        plot(vals_ss(inds_remove), 0, 'ob');
    end
end
end