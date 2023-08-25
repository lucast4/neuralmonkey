function plot_waveforms_singleplot(wf, n)
    n = min(n, size(wf,1));
    inds = randperm(size(wf, 1), n);
	plot(wf(inds,:)');
end