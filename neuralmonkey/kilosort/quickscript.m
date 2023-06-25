clear all; close all;
addpath(genpath('/gorilla1/code/neuropixel-utils'));

DIR_KILOSORT_RES = '/mnt/Freiwald/kgupta/neural_data/Pancho/220714/RSn2_batch1';
N_CHANS_PER_BATCH = 64;

if N_CHANS_PER_BATCH==64
    PATH_CHANNEL_MAP = ['/mnt/Freiwald/kgupta/neural_data/chanMap64.mat'];
elseif N_CHANS_PER_BATCH==32
    PATH_CHANNEL_MAP = ['/mnt/Freiwald/kgupta/neural_data/chanMap32.mat'];
else
    assert(false);
end

setenv('NEUROPIXEL_MAP_FILE', PATH_CHANNEL_MAP);

%%

ks = Neuropixel.KilosortDataset(DIR_KILOSORT_RES, ...
    'deduplicate_spikes', false, ...
    'deduplicate_cutoff_spikes', false);
%%
ks.load()

%% Extract waveforms
cluster_id = 1;
snippetSet = ks.getWaveformsFromRawData('cluster_ids', cluster_id, ...
    'num_waveforms', 50, 'best_n_channels', 20, 'car', true);