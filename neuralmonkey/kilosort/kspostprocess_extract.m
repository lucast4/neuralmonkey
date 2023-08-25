function kspostprocess_extract(ANIMAL, DATE)
% Given kilosorted data, run postprocessing to extract spikes across all batches,
% and save, data which can be used for automatically curating ande labeling.
% PARAMS:
% - ANIMAL, str
% - DATE, int, YYMMDD
% - LIST_BATCH, array of ints, indexing the batrches for each rsn, e,g .
% rsn2_batch1...4, then give 1:4.
% EXAMPLE:
% ANIMAL = 'Pancho';
% DATE = 230626;
% LIST_BATCH = 1:4;
% LT 8/25/23


%% MODIFY PARAMS

PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';
LOADDIR_BASE = '/mnt/Freiwald/kgupta/neural_data'; % location of kilosorted data
SAVEDIR_LOCAL = '/gorilla4/neural_preprocess_kilosort'; % fast ssd
SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/ltian/neural_data/preprocessing/kilosort_postprocess'; % final, so all machines can access.


%% old params
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

% LOADDIR_BASE = '/mnt/Freiwald/kgupta/neural_data_curated';



%% TODO
% 1) automatically extract LIST_BATCH

%% HARD CODED PARAMS

LIST_RSN = 2:3;
nWf = 500;

addpath(genpath(PATH_TO_SPIKES_CODE));
addpath(genpath(PATH_TO_NPY_CODE));
addpath(genpath([PATH_TO_KILOSORT_CODE '/postProcess']));

SAVEDIR_BASE = SAVEDIR_LOCAL;
SAVEDIR_BASE_DATE = [SAVEDIR_BASE '/' ANIMAL '/' num2str(DATE)];

SAVEDIR_FINAL_BASE = SAVEDIR_FINAL_SERVER;
SAVEDIR_FINAL = [SAVEDIR_FINAL_BASE '/final_clusters/' ANIMAL '/' num2str(DATE)];
mkdir(SAVEDIR_FINAL);


%% load globals params
[indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR] = quickscript_config_load();

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

%% SAVE SPIKE WAVEFORMS ACROSS BATCHES
for RSN = LIST_RSN
    for BATCH = LIST_BATCH
        
        myKsDir = [LOADDIR_BASE '/' ANIMAL '/' num2str(DATE) '/RSn' num2str(RSN) '_batch' num2str(BATCH)];
        SAVEDIR = [SAVEDIR_BASE_DATE '/RSn' num2str(RSN) '_batch' num2str(BATCH)];
        
        % myEventTimes = load('C:\...\data\someEventTimes.mat'); % a vector of times in seconds of some event to align to
        % Loading data from kilosort/phy easily
        
        params = struct;
        params.excludeNoise=true;
        sp = loadKSdir(myKsDir, params);
        
        disp('----------');
        disp(BATCH);
        disp(sp.cids);
        % sp.st are spike times in seconds
        % sp.clu are cluster identities
        % spikes from clusters labeled "noise" have already been omitted
        
        %% Post-processing metrics
        % 1. FIlter:
        % - low ampl spikes
        % - find spikes present in multiple clusters and just keep 1.
        % - merge all mu
        % 2. score
        % - signal to noise
        % - refrac period violations.
        
        %% GOOD - get mapping between clust and chan
        
        savedir = [SAVEDIR '/clust_chan_mappings'];
        mkdir(savedir);
        
        list_clust = sort(unique(sp.clu));
        % NOTE: this is better than sp.cids, since the later can have clusters that
        % dont exist in data. not sure why (is not because they are noise c that
        % have been removed).
        
        gwfparams.dataDir = myKsDir;    % KiloSort/Phy output folder
        % apD = dir(fullfile(myKsDir, '*ap*.bin')); % AP band file from spikeGLX specifically
        gwfparams.fileName = ['temp_wh.dat'];         % .dat file containing the raw
        gwfparams.dataType = 'int16';            % Data type of .dat file (this should be BP filtered)
        gwfparams.nCh = sp.n_channels_dat;                      % Number of channels that were streamed to disk in .dat file
        % gwfparams.wfWin = [-16 32];              % Number of samples before and after spiketime to include in waveform
        % gwfparams.nWf = 500;                    % Number of waveforms per unit to pull out
        gwfparams.wfWin = wind_spike;              % Number of samples before and after spiketime to include in waveform
        gwfparams.nWf = nWf;                    % Number of waveforms per unit to pull out
        assert(gwfparams.wfWin(1) + indpeak == 1);
        
        for i=1:length(list_clust)
            CLUST = list_clust(i);
            disp(CLUST);
            
            %     CLUSTS = [295 296];
            %     gwfparams.spikeTimes = ceil(sp.st(ismember(sp.clu, CLUSTS))*sp.sample_rate); % Vector of cluster spike times (in samples) same length as .spikeClusters
            %     gwfparams.spikeClusters = sp.clu(ismember(sp.clu, CLUSTS));
            
            gwfparams.spikeTimes = ceil(sp.st(sp.clu==CLUST)*sp.sample_rate); % Vector of cluster spike times (in samples) same length as .spikeClusters
            gwfparams.spikeClusters = sp.clu(sp.clu==CLUST);
            
            wf = getWaveForms(gwfparams);
            
            %     figure;
            %     imagesc(squeeze(wf.waveFormsMean))
            %     set(gca, 'YDir', 'normal'); xlabel('time (samples)'); ylabel('channel number');
            %     colormap(colormap_BlueWhiteRed); caxis([-1 1]*max(abs(caxis()))/2); box off;
            
            % which channel this cluster map to?
            %             tmp = abs(wf.waveFormsMean);
            %             [ampstmp] = max(tmp, [], 3);
            
            ampstmp = max(squeeze(wf.waveFormsMean), [], 2) - min(squeeze(wf.waveFormsMean), [], 2);
            [~, chan] = max(ampstmp);
            
            % sanity check that max is higher then 2nd max by a lot
            tmp = sort(ampstmp);
            if tmp(end-1)/tmp(end)<0.5
                GOOD = true;
            else
                GOOD = false;
            end
            
            % extract spikes
            waveforms = squeeze(wf.waveForms(1, :, chan, :)); % (ndat, ntimes)
            
            % SAVE
            fname = [savedir '/clust_' num2str(CLUST) '-waveforms'];
            save(fname, 'waveforms');
            %             fname = [savedir '/clust_' num2str(CLUST) '-wf'];
            %             save(fname, 'wf');
            fname = [savedir '/clust_' num2str(CLUST) '-chan'];
            save(fname, 'chan');
            fname = [savedir '/clust_' num2str(CLUST) '-GOOD'];
            save(fname, 'GOOD');
            
            % Save which cluster group it was given.
            clust_group_id = sp.cgs(sp.cids==CLUST);
            % - 0 = noise
            % - 1 = mua
            % - 2 = good
            % - 3 = unsorted
            if isempty(clust_group_id)
                % This is possible, somehow this clust was not
                % assigned a label
                clust_group_name = 'unknown';
            else
                switch clust_group_id
                    case 0
                        clust_group_name = 'noise';
                    case 1
                        clust_group_name = 'mua';
                    case 2
                        clust_group_name = 'good';
                    case 3
                        clust_group_name = 'unsorted';
                    otherwise
                        assert(false);
                end
            end
            fname = [savedir '/clust_' num2str(CLUST) '-clust_group_name'];
            save(fname, 'clust_group_name');
            fname = [savedir '/clust_' num2str(CLUST) '-clust_group_id'];
            save(fname, 'clust_group_id');
            
            % Actual times of all spikes (sec)
            times_sec_all = sp.st(sp.clu==CLUST);
            fname = [savedir '/clust_' num2str(CLUST) '-times_sec_all'];
            save(fname, 'times_sec_all');
        end
        
    end
end
end