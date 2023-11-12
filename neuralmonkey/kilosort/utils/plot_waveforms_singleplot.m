function plot_waveforms_singleplot(wf, n, remove_outliers)
% Plot on gcf these waveforms, without any preprocessing 
% except remove outlier.
% PARAMS
% - wf, (ntrials, timebins)

if ~exist('remove_outliers', 'var'); remove_outliers=true; end

if remove_outliers
	[wf] = waveforms_remove_outliers(wf, false);
end

XLIM = [0 50];

n = min(n, size(wf,1));
inds = randperm(size(wf, 1), n);
plot(wf(inds,:)');

% overlay mean
wf_mean = mean(wf(inds,:), 1);
plot(wf_mean, '-k', 'LineWidth', 1.5);

xlim(XLIM);
YLIM = ylim();
% overlay text for sample size
YDELT = YLIM(2)-YLIM(1);
XDELT = XLIM(2)-XLIM(1);
text(XLIM(1)+XDELT*0.1, YLIM(1)+YDELT*0.1, ['n=' num2str(n)]);
end