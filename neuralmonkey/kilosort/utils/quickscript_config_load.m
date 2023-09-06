function [indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR, SNR_VER] = quickscript_config_load()
	% Load global defualt params


	wind_spike = [-17 38]; % samples pre and post
	indpeak = -wind_spike(1) + 1; % index of peak of spike.
    
    npre = 9; % window for cmputing snr, in samples, rel spike peak.
    npost = 20;
    
    % For classifying
    if false
        % - Using MEDIAN version of snr compute
        SNR_VER = 'median';
        THRESH_SU_SNR = 5;
        MIN_SNR = 2.25;
    else
        % - Using MEAN version of snr compute
        SNR_VER = 'mean';
        THRESH_SU_SNR = 8;
        MIN_SNR = 4.2;
    end

    THRESH_SU_ISI = 0.02;
    THRESH_ARTIFACT_SHARP = 16;
    THRESH_ARTIFACT_SHARP_LOW = 10;
    THRESH_ARTIFACT_ISI = 0.12;

end
