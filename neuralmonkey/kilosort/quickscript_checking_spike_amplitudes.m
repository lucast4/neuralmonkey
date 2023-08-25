%% Quickly comparing two methods for gerting actual amlitudes: (i) using template x scaling, (ii) using waveform.

% 1) Get amplitudes for each spike
[spikeAmps, spikeDepths, templateYpos, tempAmps, tempsUnW, tempDur, tempPeakWF] = ...
    templatePositionsAmplitudes(sp.temps, sp.winv, sp.ycoords, sp.spikeTemplates, sp.tempScalingAmps);

%%
sp.spikeAmps = spikeAmps;
sp.spikeDepths = spikeDepths;
sp.templateYpos = templateYpos;
sp.tempAmps = tempAmps;
sp.tempsUnW = tempsUnW;
sp.tempDur = tempDur;
sp.tempPeakWF = tempPeakWF;

%% 2) extract waveforms

CLUST = 81;

gwfparams.dataDir = myKsDir;    % KiloSort/Phy output folder
% apD = dir(fullfile(myKsDir, '*ap*.bin')); % AP band file from spikeGLX specifically
gwfparams.fileName = ['temp_wh.dat'];         % .dat file containing the raw
gwfparams.dataType = 'int16';            % Data type of .dat file (this should be BP filtered)
gwfparams.nCh = sp.n_channels_dat;                      % Number of channels that were streamed to disk in .dat file
% gwfparams.wfWin = [-16 32];              % Number of samples before and after spiketime to include in waveform
% gwfparams.nWf = 500;                    % Number of waveforms per unit to pull out
gwfparams.wfWin = [-16 32];              % Number of samples before and after spiketime to include in waveform
gwfparams.nWf = 500;                    % Number of waveforms per unit to pull out

assert(gwfparams.wfWin(1) + indpeak == 1);

gwfparams.spikeTimes = ceil(sp.st(sp.clu==CLUST)*sp.sample_rate); % Vector of cluster spike times (in samples) same length as .spikeClusters
gwfparams.spikeClusters = sp.clu(sp.clu==CLUST);

wf = getWaveForms(gwfparams);

%% [ignore] plot mean waveforms over all chans for this clust

figure; hold on;
imagesc(squeeze(wf.waveFormsMean(1, :, :)));

figure; hold on;
waveforms = squeeze(wf.waveForms(1, :, 19, :));
plot(waveforms(1:100,:)')

figure; hold on;
waveforms = squeeze(wf.waveForms(1, :, 21, :));
plot(waveforms(1:100,:)')

%% 3) confirm that amplitudes match waveforms/200
ampstmp = max(squeeze(wf.waveFormsMean), [], 2) - min(squeeze(wf.waveFormsMean), [], 2);
[~, chan] = max(maxvals);

amps = sp.spikeAmps(sp.clu==CLUST);
amps2 = squeeze(max(wf.waveForms(1,:,chan,:), [], 4) - min(wf.waveForms(1,:,chan,:), [], 4)); % (1, ndat)

mean(amps)
mean(amps2)

%% Conclusion
% NOTE: these give similar values.
% NOTE: these are about 15x higher than uV (based on eyeballing relative to
% waveforms using tdt).
