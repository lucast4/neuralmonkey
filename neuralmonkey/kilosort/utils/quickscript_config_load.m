function [indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR, SNR_VER, THRESH_SU_SNR_LOWER, THRESH_SU_SNR_HIGHEST] = quickscript_config_load()
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
        assert(false, 'determine what to use for THRESH_SU_SNR_LOWER')
    else
        % - Using MEAN version of snr compute
        SNR_VER = 'mean';
        % THRESH_SU_SNR = 8;
        THRESH_SU_SNR = 7.9; % 11/12/23 , more leneient
        % THRESH_SU_SNR_LOWER = 7;
        THRESH_SU_SNR_LOWER = 6.9; % 11/12/23 , more leneient
        % THRESH_SU_SNR_HIGHEST = 10;
        THRESH_SU_SNR_HIGHEST = 9.6; % 11/12/23 , more leneient.
%         MIN_SNR = 4.1;
        % MIN_SNR = 4.2; % 9/17/23 - updated, to get more MU.
        % MIN_SNR = 4.025; % 9/22/23 - updated, I think I did accidentally on 9/17
        MIN_SNR = 3.9; % 11/12/23 - updated to be more lenient
    end

    THRESH_SU_ISI = 0.02;
    THRESH_ARTIFACT_SHARP = 16;
    THRESH_ARTIFACT_SHARP_LOW = 10;
    THRESH_ARTIFACT_ISI = 0.12;

end
