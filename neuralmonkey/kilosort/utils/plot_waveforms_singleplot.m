function plot_waveforms_singleplot(wf, n)
% Plot on gcf these waveforms,
% PARAMS
% - wf, (ntrials, timebins)

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