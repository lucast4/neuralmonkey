function kspostprocess_metrics_and_label(ANIMAL, DATE)
% given data that has been kilosorted, and extracred using quickscript_spikes_extract
% run this to curate and autoamticlaly label data.
% Runs metrics, labels each cluster, then saves plots and guis that can
% load later to manually curate + merge into final clusters.


%% MODIFY PARAMS

PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';

LOADDIR_BASE = '/lemur2/kilosort_data'; % location of kilosorted data
SAVEDIR_LOCAL = '/lemur2/kilosort_temp'; % fast ssd
SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/kgupta/neural_data/postprocess'; % final, so all machines can access.

%% old params

% ANIMAL = 'Pancho';
%
% DATE = 230626;
% LIST_BATCH = 1:4;

% DATE = 220715;
% LIST_BATCH = 1:8;

% DATE = 230620;
% LIST_BATCH = 1:4;

% DATE = 230620;
% LIST_BATCH = 1:4;

% DATE = 220621;
% LIST_BATCH = 1:4;

% DATE = 220716;
% LIST_BATCH = 1:4;

%% HARD CODED PARAMS

LIST_RSN = 2:3;
% nWf = 500;

addpath(genpath(PATH_TO_SPIKES_CODE));
addpath(genpath(PATH_TO_NPY_CODE));
addpath(genpath([PATH_TO_KILOSORT_CODE '/postProcess']));

SAVEDIR_BASE = SAVEDIR_LOCAL;
SAVEDIR_BASE_DATE = [SAVEDIR_BASE '/' ANIMAL '/' num2str(DATE)];

SAVEDIR_FINAL_BASE = SAVEDIR_FINAL_SERVER;
SAVEDIR_FINAL = [SAVEDIR_FINAL_BASE '/final_clusters/' ANIMAL '/' num2str(DATE)];

%% LOAD GLOBAL PARAMS

[indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR] = quickscript_config_load();

%% Chek that prev step done

path =[SAVEDIR_FINAL '/DONE_kspostprocess_extract.mat'];
assert(exist(path, 'file')==2, ['Did not complete prev step, ' path]);

%% Get LIST_BATCH automatically.
% e.g, LIST_BATCH = 1:4, meaning you have rsn2_batch1...4 and rsn3_batch1...4

list_batch_tmp = [];
for RSN = LIST_RSN
    tmp = [LOADDIR_BASE '/' ANIMAL '/' num2str(DATE) '/RSn' num2str(RSN) '_batch*'];
    dirsthis = dir(tmp);
    list_batch_tmp(end+1) = length(dirsthis);
end


nbatch = unique(list_batch_tmp);
assert(length(nbatch)==1);
LIST_BATCH = 1:nbatch;

% sanity check that all dirs exist
for RSN = LIST_RSN
    for BATCH = LIST_BATCH
        myKsDir = [LOADDIR_BASE '/' ANIMAL '/' num2str(DATE) '/RSn' num2str(RSN) '_batch' num2str(BATCH)];
        assert(exist(myKsDir, 'dir')==7);
    end
end

disp('Got this LIST_BATCH:');
disp(LIST_BATCH)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOAD PRE-SAVED DATA ACROSS BATCHES

DATSTRUCT = [];
for RSN = LIST_RSN
    for BATCH = LIST_BATCH
        
        SAVEDIR = [SAVEDIR_BASE_DATE '/RSn' num2str(RSN) '_batch' num2str(BATCH) '/clust_chan_mappings'];
        
        disp(SAVEDIR);
        
        % Load each cluster
        files = dir(SAVEDIR);
        list_clust_loaded = [];
        for i=1:length(files)
            fname = files(i).name; % clust_9-wf.mat
            if length(fname)<3
                continue
            end
            ind1 = strfind(fname, '-');
            %             ind2 = strfind(fname, '.mat');
            
            clust = str2num(fname(7:ind1-1));
            var = fname(ind1+1:end-4);
            
            if strcmp(var, 'wf')
                % Skip, since is too large.
                continue
            end
            
            dat = load([files(i).folder '/' fname]);
            
            DATSTRUCT = [DATSTRUCT, ...
                struct('clustnum', clust, 'var', var, 'dat', dat.(var), 'batch', BATCH, 'RSn', RSN)];
            
            list_clust_loaded(end+1) = clust;
        end
        
        % CHECK previous step was done...
        % sanity checxk that got all clusters. (i.e., did not crash during saving.)
        path = [SAVEDIR_BASE_DATE '/RSn' num2str(RSN) '_batch' num2str(BATCH) '/list_clust.mat'];
        tmp = load(path);
        list_clust_exist = sort(unique(tmp.list_clust))';
        list_clust_loaded = sort(unique(list_clust_loaded));
        a = all(ismember(list_clust_loaded, list_clust_exist));
        b = all(ismember(list_clust_exist, list_clust_loaded));
        if ~a | ~b
            disp(list_clust_loaded);
            disp(list_clust_exist);
            assert(false, 'BAD! missing clusters');
        else
            disp('GOOD! loaded all clusters');
        end
    end
end

% Reshape so each datapt is a single cluster
DATSTRUCT2 = [];
list_var = {'chan', 'clust_group_id', 'clust_group_name', 'GOOD', 'times_sec_all', 'waveforms'};

for RSN = LIST_RSN
    for BATCH = LIST_BATCH
        
        disp([RSN, BATCH]);
        inds = [DATSTRUCT.batch]==BATCH & [DATSTRUCT.RSn]==RSN;
        list_clust = sort(unique([DATSTRUCT(inds).clustnum]));
        
        for j=1:length(list_clust)
            clust = list_clust(j);
            dat = struct;
            dat.batch = BATCH;
            dat.RSn = RSN;
            dat.clust = clust;
            for i=1:length(list_var)
                var = list_var{i};
                
                inds = [DATSTRUCT.clustnum]==clust & [DATSTRUCT.batch]==BATCH & [DATSTRUCT.RSn]==RSN & strcmp({DATSTRUCT.var}, var);
                assert(sum(inds)==1)
                dat.(var) = DATSTRUCT(inds).dat;
            end
            
            DATSTRUCT2 = [DATSTRUCT2, dat];
        end
    end
end

DATSTRUCT = DATSTRUCT2;
clear DATSTRUCT2

% Print all [rsn batch clust]
% [[DATSTRUCT.RSn]' [DATSTRUCT.batch]' [DATSTRUCT.chan]' [DATSTRUCT.clust]']

%% convert to global channel
nbatches = length(LIST_BATCH);
nchans_per_batch = 256/nbatches;
ct = 1;
chan_mapper = nan(3, nbatches, nchans_per_batch); % (rs, batch, chan_within_batch) --> chan_global.
for rs=2:3
    for batch=LIST_BATCH
        disp([rs, batch]);
        chans_withinbatch = 1:nchans_per_batch;
        chans_global = ct:ct+nchans_per_batch-1;
        ct=ct+nchans_per_batch;
        %         disp(chansthis)
        
        chan_mapper(rs, batch, chans_withinbatch) = chans_global;
    end
end

% assign to datstructu
for i=1:length(DATSTRUCT)
    chan = chan_mapper(DATSTRUCT(i).RSn, DATSTRUCT(i).batch, DATSTRUCT(i).chan);
    assert(~isnan(chan));
    DATSTRUCT(i).chan_global = chan;
end

%% OTher cleanup of datstruct
for i=1:length(DATSTRUCT)
    % save index
    DATSTRUCT(i).index = i;
    
    % remove nans from waverforms
    wf = DATSTRUCT(i).waveforms;
    wf(isnan(wf(:,1)), :) = [];
    DATSTRUCT(i).waveforms = wf;
    
    % Correct shape
    if size(DATSTRUCT(i).times_sec_all, 1)==1
        DATSTRUCT(i).times_sec_all = DATSTRUCT(i).times_sec_all';
    end
    
    % get amplitudes
    wf = DATSTRUCT(i).waveforms;
    amps = max(wf, [], 2) - min(wf, [], 2);
    DATSTRUCT(i).amps_wf = amps;
end

%% Find double-counted waveforms
% if refrac violations are really high within short time...
DRYRUN = false;
DATSTRUCT = datstruct_remove_double_counted(DATSTRUCT, indpeak, npre, npost, DRYRUN);

% To pick out a sprcific one
% datstruct_remove_double_counted(DATSTRUCT(1354), indpeak, npre, npost, DRYRUN);

%% Assign metrics and save plots of waveforms along with snr
DOPLOT = true; % Might takes a while...
DATSTRUCT = datstruct_compute_metrics(DATSTRUCT, DOPLOT, ...
    indpeak, npre, npost, SAVEDIR_FINAL);

%% ASSIGNING SU
% 1) if bipolar, then not SU.
% 2) snr>4.4 and Q<0.05
% - clear cases:
% 4.9,

% TODO:
% - compute traditional refractoriness.

% NOTES ON THRESHOLDS:
% - snr>4.9
% - Q<0.05;
% - isiviolations (frac)<0.03

% THRESH_SU_SNR = 5;
% THRESH_SU_ISI = 0.02;
% THRESH_ARTIFACT_SHARP = 20;
% THRESH_ARTIFACT_SHARP_LOW = 10;
% THRESH_ARTIFACT_ISI = 0.12;
% MIN_SNR = 2.25;

DATSTRUCT = datstruct_classify(DATSTRUCT, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR);

% %%
% DATSTRUCT_FINAL = datstruct_merge(DATSTRUCT, SAVEDIR_FINAL, ...
%     LIST_MERGE_SU, indpeak, npre, npost, THRESH_SU_SNR, ...
%     THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
%     THRESH_ARTIFACT_ISI, MIN_SNR);
%
% %% Recompute metrics on merged data
%
% DOPLOT = false;
% DATSTRUCT_FINAL = datstruct_compute_metrics(DATSTRUCT_FINAL, DOPLOT, ...
%     indpeak, npre, npost);


%% SAVE FINAL for this day, each clustre, not yet merged.
% save([SAVEDIR_FINAL '/DATSTRUCT_FINAL.mat'], 'DATSTRUCT_FINAL'); % donrt
% save. will have to clean
% save([SAVEDIR_FINAL '/DATSTRUCT.mat'], 'DATSTRUCT', '-v7.3');
datstruct_save(DATSTRUCT, SAVEDIR_FINAL, '');

%% Make summary figures
close all;
datstruct_plot_summary(DATSTRUCT, SAVEDIR_FINAL);

%% ###################################
%% ############## PLOT WAVEFORMS ORDERED BY METRICS.
if false % wait until after curation to do this.
    close all;
    datstruct_plot_waveforms_all(DATSTRUCT, SAVEDIR_FINAL, THRESH_SU_SNR, ...
        THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
        THRESH_ARTIFACT_ISI, MIN_SNR);
end

%% LOAD DATSTRUCT

%% [GUI] SAVE PLOTS for later manual curation.
MAKE_GUI = true;
savedir = [SAVEDIR_FINAL '/waveform_gridplots_GUI'];
mkdir(savedir);

% 1) NOISE
close all;
path_noext = [savedir '/noise-sorted_by_snr'];
values = [DATSTRUCT.snr_final];
exclude_labels = {'su', 'mua', 'artifact'};
values_sort = [MIN_SNR-0.75, MIN_SNR+0.6];
YLIM = [-2200 1000];
map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
    path_noext, exclude_labels, false, MAKE_GUI, YLIM);

% 1) MU
close all;
path_noext = [savedir '/MU-sorted_by_snr'];
values = [DATSTRUCT.snr_final];
exclude_labels = {'su', 'noise', 'artifact'};
values_sort = [MIN_SNR-0.5, MIN_SNR+1];
YLIM = [-2200 1000];
map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
    path_noext, exclude_labels, false, MAKE_GUI, YLIM);

close all;
path_noext = [savedir '/MU-ALL'];
values = [DATSTRUCT.snr_final];
exclude_labels = {'su', 'noise', 'artifact'};
values_sort = [MIN_SNR-0.5, MIN_SNR+100000];
YLIM = [-2200 1000];
map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
    path_noext, exclude_labels, false, MAKE_GUI, YLIM);

% All SU
close all;
path_noext = [savedir '/SU-sorted_by_snr'];
values = [DATSTRUCT.snr_final];
exclude_labels = {'noise', 'mua', 'artifact'};
values_sort = [];
YLIM = [-4000, 1500];
map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
    path_noext, exclude_labels, false, MAKE_GUI, YLIM);

% % MU-SU boundary
% close all;
% path_noext = [savedir '/snr-mu_su_boundary'];
% values = [DATSTRUCT.snr_final];
% values_sort =  [THRESH_SU_SNR-1, THRESH_SU_SNR+1000];
% exclude_labels = {'noise'};
% map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
%     path_noext, exclude_labels, false, MAKE_GUI);

% artifacts
close all;
path_noext = [savedir '/artifact-sorted_by_sharpiness'];
values = [DATSTRUCT.sharpiness];
exclude_labels = {'noise', 'mua', 'su'};
values_sort = [];
map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
    path_noext, exclude_labels, false, MAKE_GUI);

% 1) su, changlobal.
%
% title with chan num.

% - Only chans with >1 SU. find indices
list_chan_global = unique([DATSTRUCT.chan_global]);
INDICES_PLOT = [];
for chan = list_chan_global
    inds = find([DATSTRUCT.chan_global]==chan);
    good = sum(strcmp({DATSTRUCT(inds).label_final}, 'su'))>1;
    
    if good
        INDICES_PLOT = [INDICES_PLOT, inds];
    end
end

if false
    close all;
    path_noext = [savedir '/SU-sorted_by_changlobal'];
    values = [DATSTRUCT.chan_global];
    exclude_labels = {'noise', 'mua', 'artifact'};
    values_sort = [];
    YLIM = [-4500, 2000];
    map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values, values_sort, ...
        path_noext, exclude_labels, false, MAKE_GUI, YLIM, INDICES_PLOT, 3, 5);
end 

%% save a note to mark done.
tmp = [];
save([SAVEDIR_FINAL '/DONE_kspostprocess_metrics_and_label.mat'], 'tmp');

%% #######################################
%% ################ SCRATCH

if false
    %% give 2 su clusters, compare them
    
    i1 = 275;
    i2 = 282;
    s1 = DATSTRUCT(i1).times_sec_all;
    s2 = DATSTRUCT(i2).times_sec_all;
    
    dt = 1/1000;
    [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
    Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
    R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
    
    %             if Q<.2 && R<.05 % if both refractory criteria are met
    %     if Q12<.25 && R<.05 % if both metrics are below threshold.
    
    figure; hold on;
    plot(K)
    line(xlim, [0 0]);
    
    figure; hold on;
    plot(Qi)
    line(xlim, [0 0]);
    
    
    %%
    
    i1 = 245;
    i2 = 248;
    s1 = DATSTRUCT(i1).times_sec_all;
    s2 = DATSTRUCT(i2).times_sec_all;
    
    % disp('----');
    % [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
    %   frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s1)
    %
    % disp('----');
    % [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
    %   frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s2)
    % %
    % % [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
    % %     frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s1)
    %
    % disp('----');
    % s12 = sort([s1; s2]);
    % [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
    %   frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s12)
    
    disp('----');
    [double_count_rate, ~, ~, ...
        frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s1)
    
    disp('----');
    [double_count_rate, ~, ~, ...
        frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s2)
    %
    % [double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
    %   frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s1)
    
    disp('----');
    s12 = sort([s1; s2]);
    [double_count_rate, ~, ~, ...
        frac_spikes_double_counted] = datstruct_remove_double_counted_inner(s12)
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Preprocessing each individual cluster
    
    ind = 49;
    
    figure; hold on;
    plot(DATSTRUCT(ind).waveforms');
    
    
    %% get snr
    
    waveforms = DATSTRUCT(ind).waveforms;
    
    indpeak = -gwfparams.wfWin(1);
    snr = snr_compute(waveforms, indpeak);
    
    %%
    
    i = 49;
    waveforms = DATSTRUCT(i).waveforms;
    [waveforms_aligned, indpeak_new] = get_shifted_wf(waveforms, ispos, indpeak, ...
        npre, npost);
    
    close all;
    
    figure;
    plot(waveforms');
    figure;
    plot(waveforms_aligned');
    
    %% comptue for each one
    list_Q = [];
    list_isiv = [];
    for i=1:length(DATSTRUCT)
        st = DATSTRUCT(i).times_sec_all;
        [Q, R, isi_violation_pct] = refractoriness_compute(st);
        isi_violation_pct_v2 = ISIViolations(st, 0, 0.002);
        
        %     if (Q>0.05 & isi_violation_pct<0.04) | (Q<0.05 & isi_violation_pct>0.04)
        disp([i, Q, R, isi_violation_pct, isi_violation_pct_v2]);
        %     end
        
        list_Q(end+1) = Q;
        list_isiv(end+1) = isi_violation_pct;
    end
    
    figure; hold on;
    plot(list_isiv, list_Q, 'ok');
    xlabel('isi violations');
    ylabel('Q');
    
    %%
    
    
    %% refrac period violations
    
    % i = 366;
    i = 49;
    
    st = DATSTRUCT(i).times_sec_all;
    
    [Q, R, isi_violation_pct] = refractoriness_compute(st, true)
    isi_violation_pct_v2 = ISIViolations(st, 0, 0.002)
    
    
    % figure; plot(st, ones(length(st)), 'ok');
    %%
    wf = DATSTRUCT(i).waveforms;
    snr_compute_wrapper(wf, indpeak, npre, npost)
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% NOTES
    
    % High refrac rate and isi violations
    % 282.
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SCRATCH
    %% bimodality (both pos and neg)
    i = 47
    
    vals
    
    figure;
    plot(DATSTRUCT(i).waveforms');
    
    for i=1:length(DATSTRUCT)
        
        wf = DATSTRUCT(i).waveforms;
        
        vals = wf(:, indpeak);
        
        %     get_shifted_wf(wf,
        
        [BF, BC] = bimodalitycoeff(vals);
        
        npos = sum(vals>0);
        nneg = sum(vals<0);
        frac_pos = npos/(npos+nneg);
        if BC>0.3 & frac_pos>0.1 & frac_pos<0.9
            disp([i, frac_pos, BC]);
        end
    end
    
    % Split into pos and negative wf
    wf_pos = wf(vals>0, :);
    wf_neg = wf(vals<=0, :);
    
    figure; hold on;
    
    subplot(1,3,1);
    plot(wf');
    subplot(1,3,2);
    plot(wf_pos');
    subplot(1,3,3);
    plot(wf_neg');
    
    % score pos
    [snr] = snr_compute_wrapper(wf, indpeak, npre, npost)
    [snr] = snr_compute_wrapper(wf_pos, indpeak, npre, npost)
    [snr] = snr_compute_wrapper(wf_neg, indpeak, npre, npost)
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% FUNCTIONS
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Loading raw waveforms
    
    CLUST = 101;
    % To get the true waveforms of the spikes (not just kilosort's template
    % shapes), use the getWaveForms function:
    
    gwfparams.dataDir = myKsDir;    % KiloSort/Phy output folder
    % apD = dir(fullfile(myKsDir, '*ap*.bin')); % AP band file from spikeGLX specifically
    gwfparams.fileName = ['temp_wh.dat'];         % .dat file containing the raw
    gwfparams.dataType = 'int16';            % Data type of .dat file (this should be BP filtered)
    gwfparams.nCh = sp.n_channels_dat;                      % Number of channels that were streamed to disk in .dat file
    gwfparams.wfWin = [-25 25];              % Number of samples before and after spiketime to include in waveform
    gwfparams.nWf = 100;                    % Number of waveforms per unit to pull out
    gwfparams.spikeTimes = ceil(sp.st(sp.clu==CLUST)*sp.sample_rate); % Vector of cluster spike times (in samples) same length as .spikeClusters
    gwfparams.spikeClusters = sp.clu(sp.clu==CLUST);
    
    wf = getWaveForms(gwfparams);
    
    figure;
    imagesc(squeeze(wf.waveFormsMean))
    set(gca, 'YDir', 'normal'); xlabel('time (samples)'); ylabel('channel number');
    colormap(colormap_BlueWhiteRed); caxis([-1 1]*max(abs(caxis()))/2); box off;
    
    %% which channel this cluster map to?
    
    tmp = abs(wf.waveFormsMean);
    [maxvals] = max(tmp, [], 3);
    [~, chan] = max(maxvals);
    
    % sanity check that max is higher then 2nd max by a lot
    tmp = sort(maxvals);
    assert(tmp(end-1)/tmp(end)<0.25)
    
    %% extract spikes
    waveforms = squeeze(wf.waveForms(1, :, chan, :)); % (ndat, ntimes)
    
    
    %% Plot overlay of all spikes
    figure; hold on;
    plot(waveforms')
    
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% OTHER STUFF
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Plotting a driftmap
    
    [spikeTimes, spikeAmps, spikeDepths, spikeSites] = ksDriftmap(myKsDir);
    figure; plotDriftmap(spikeTimes, spikeAmps, spikeDepths);
    
    %% basic quantification of spiking plot
    
    depthBins = 0:40:3840;
    ampBins = 0:30:min(max(spikeAmps),800);
    recordingDur = sp.st(end);
    
    [pdfs, cdfs] = computeWFampsOverDepth(spikeAmps, spikeDepths, ampBins, depthBins, recordingDur);
    plotWFampCDFs(pdfs, cdfs, ampBins, depthBins);
    
    
    %% Plotting some basics about LFPs
    if false
        lfpD = dir(fullfile(myKsDir, '*.lf.bin')); % LFP file from spikeGLX specifically
        lfpFilename = fullfile(myKsDir, lfpD(1).name);
        
        lfpFs = 2500;  % neuropixels phase3a
        nChansInFile = 385;  % neuropixels phase3a, from spikeGLX
        
        [lfpByChannel, allPowerEst, F, allPowerVar] = ...
            lfpBandPower(lfpFilename, lfpFs, nChansInFile, []);
        
        chanMap = readNPY(fullfile(myKsDir, 'channel_map.npy'));
        nC = length(chanMap);
        
        allPowerEst = allPowerEst(:,chanMap+1)'; % now nChans x nFreq
        
        % plot LFP power
        dispRange = [0 100]; % Hz
        marginalChans = [10:50:nC];
        freqBands = {[1.5 4], [4 10], [10 30], [30 80], [80 200]};
        
        plotLFPpower(F, allPowerEst, dispRange, marginalChans, freqBands);
    end
    
    %% Computing some useful details about spikes/neurons (like depths)
    
    [spikeAmps, spikeDepths, templateYpos, tempAmps, tempsUnW, tempDur, tempPeakWF] = ...
        templatePositionsAmplitudes(sp.temps, sp.winv, sp.ycoords, sp.spikeTemplates, sp.tempScalingAmps);
    
    %% load synchronization data
    if false
        syncChanIndex = 385;
        syncDat = extractSyncChannel(myKsDir, nChansInFile, syncChanIndex);
        
        eventTimes = spikeGLXdigitalParse(syncDat, lfpFs);
        
        % - eventTimes{1} contains the sync events from digital channel 1, as three cells:
        % - eventTimes{1}{1} is the times of all events
        % - eventTimes{1}{2} is the times the digital bit went from off to on
        % - eventTimes{1}{2} is the times the digital bit went from on to off
        
        % To make a timebase conversion, e.g. between two probes:
        % [~,b] = makeCorrection(syncTimesProbe1, syncTimesProbe2, false);
        
        % and to apply it:
        % correctedSpikeTimes = applyCorrection(spikeTimesProbe2, b);
    end
    
    %% Looking at PSTHs aligned to some event
    
    eventTimes = [0.01 0.02]
    % if you now have a vector of relevant event times, called eventTimes (but
    % not the cell array as above, just a vector):
    
    window = [-0.3 1]; % look at spike times from 0.3 sec before each event to 1 sec after
    
    % if your events come in different types, like different orientations of a
    % visual stimulus, then you can provide those values as "trial groups",
    % which will be used to construct a tuning curve. Here we just give a
    % vector of all ones.
    trialGroups = ones(size(eventTimes));
    
    psthViewer(sp.st, sp.clu, eventTimes, window, trialGroups);
    
    % use left/right arrows to page through the clusters
    
    
    %% PSTHs across depth
    
    depthBinSize = 80; % in units of the channel coordinates, in this case µm
    timeBinSize = 0.01; % seconds
    bslWin = [-0.2 -0.05]; % window in which to compute "baseline" rates for normalization
    psthType = 'norm'; % show the normalized version
    eventName = 'stimulus onset'; % for figure labeling
    
    [timeBins, depthBins, allP, normVals] = psthByDepth(spikeTimes, spikeDepths, ...
        depthBinSize, timeBinSize, eventTimes, window, bslWin);
    
    figure;
    plotPSTHbyDepth(timeBins, depthBins, allP, eventName, psthType);
    
end
end