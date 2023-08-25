function [indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR] = quickscript_config_load()
	% Load global defualt params


	wind_spike = [-17 38]; % samples pre and post
	indpeak = -wind_spike(1) + 1; % index of peak of spike.
    
    npre = 9; % window for cmputing snr, in samples, rel spike peak.
    npost = 20;
    
    % For classifying
    THRESH_SU_SNR = 5;
    THRESH_SU_ISI = 0.02;
    THRESH_ARTIFACT_SHARP = 20;
    THRESH_ARTIFACT_SHARP_LOW = 10;
    THRESH_ARTIFACT_ISI = 0.12;
    MIN_SNR = 2.25;

end
