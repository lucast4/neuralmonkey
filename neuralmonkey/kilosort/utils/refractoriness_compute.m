function [Q, R, isi_violation_pct] = refractoriness_compute(st, ploton)
	% RETURNS:
	% - Q, refractoriness, relative prob of spike at close lag
	% compoared to far lag.

	if ~exist('ploton', 'var'); ploton=false; end

	dt = 0.001;
	[K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt);
	Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
	R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes

	% Do it the dumb version for comparison
	threshold = 0.0015; % second, to call it a violatioin.
	st = sort(st);
    isi = diff(st);
    violations = sum(isi <= threshold);
    num_isi = length(isi);
    isi_violation_pct = violations/num_isi; 

	if ploton
		figure; hold on;
		plot(K)
		figure; hold on;
		plot(Qi)
		% plot(Q00)
		% plot(Q01)
		disp(['Contamination rate: ', num2str(Q)]); % want < 0.1
		disp(['Refractorinessv2: ', num2str(R)]); % want < 0.05
	end
end