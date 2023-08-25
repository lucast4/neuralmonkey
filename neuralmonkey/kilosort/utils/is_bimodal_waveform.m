function [isbimod, waveforms_pos, waveforms_neg] = is_bimodal_waveform(waveforms, indpeak)
	% Conservative, so if there is hint of biodal, returns true.
	% waveforms, (ndat, timebins)
    vals = waveforms(:, indpeak);
    [BF, BC] = bimodalitycoeff(vals);
    npos = sum(vals>0);
    nneg = sum(vals<0);
    frac_pos = npos/(npos+nneg);
    isbimod = BC>0.3 & frac_pos>0.1 & frac_pos<0.9;

    waveforms_pos = waveforms(vals>0, :);
	waveforms_neg = waveforms(vals<=0, :);

	if false
		figure; hold on;

		subplot(1,3,1);
		plot(waveforms');
		subplot(1,3,2);
		plot(waveforms_pos');
		subplot(1,3,3);
		plot(waveforms_neg');
	end
end