function ispositive = is_positive_waveform(waveforms)
	% waveforms, (ndat, timebins)
	wf_mean = mean(waveforms, 1);
    ispositive = abs(max(wf_mean))>abs(min(wf_mean)); % whether pos or neg spike
end