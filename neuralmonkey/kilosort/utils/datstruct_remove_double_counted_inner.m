function [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
	frac_spikes_double_counted] = datstruct_remove_double_counted_inner(st, ...
		THRESH_CLOSE)

	%% Find double-counted waveforms
	% if refrac violations are really high within short time...
	if ~exist('THRESH_CLOSE', 'var'); THRESH_CLOSE = 0.00025; end
		
    threshold = 0.0005; % second, to call it a violatioin.

    st = sort(st);
    isi = diff(st);

    % THRESH_CLOSE = 0.00025;
    THRESH_FAR = 0.002;
    THRESH_BASE = 0.02;
    violations_close = isi<THRESH_CLOSE;
    violations_far = (isi<THRESH_FAR) & (isi>THRESH_CLOSE);
    violations_base = (isi<THRESH_BASE) & (isi>THRESH_FAR);

    rate_close = sum(violations_close)/(THRESH_CLOSE - 0);
    rate_far = sum(violations_far)/(THRESH_FAR - THRESH_CLOSE);
    rate_base = sum(violations_base)/(THRESH_BASE - THRESH_FAR);
    
    if rate_far==0
        rate_far = rate_base;
    end
    
    double_count_rate = rate_close/rate_far;
    double_count_rate_relbase = rate_close/rate_base;
    rate_far_rel_base = rate_far/rate_base;
    
    % estimate frac of total spiies that are double counted
    n_double = sum(violations_close);
    n_total = length(st);
    frac_spikes_double_counted = n_double/n_total;

