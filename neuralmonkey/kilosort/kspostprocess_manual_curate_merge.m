function kspostprocess_manual_curate_merge(ANIMAL, DATE)
% given data that has been kilosorted, and extracred using quickscript_spikes_extract
% run this to curate and autoamticlaly label data.


%% MODIFY PARAMS

PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';
% LOADDIR_BASE = '/mnt/Freiwald/kgupta/neural_data'; % location of kilosorted data
SAVEDIR_LOCAL = '/gorilla4/neural_preprocess_kilosort'; % fast ssd
SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/ltian/neural_data/preprocessing/kilosort_postprocess'; % final, so all machines can access.

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

% LIST_RSN = 2:3;
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

%% Load datstruct, with metrics already computed (ther ouptut of quickscript_spikes)

tmp = load([SAVEDIR_FINAL '/DATSTRUCT.mat']);
DATSTRUCT = tmp.DATSTRUCT;
clear tmp

%% [LOAD GUI] and manually curate


savedir = [SAVEDIR_FINAL '/waveform_gridplots_GUI'];

DATSTRUCT_MOD = DATSTRUCT;

% STEPS:

%%% (1) noise, add those that are MU
figpath = 'noise-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'cancel');
assert_current_label = 'noise';
[DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
    savedir, instructions, assert_current_label);

%%% (2) MU, remove those that are noise
figpath = 'MU-sorted_by_snr';
instructions = struct('left', 'to_noise', 'right', 'cancel');
assert_current_label = 'mua';

[DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
    savedir, instructions, assert_current_label);

% 2. SU, remove that that are MU.
figpath = 'SU-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'to_artifact');
assert_current_label = 'su';

[DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
    savedir, instructions, assert_current_label);

% 3. artifact, keep those that are SU.
figpath = 'artifact-sorted_by_sharpiness';
instructions = struct('left', 'to_su', 'right', 'to_mua');
assert_current_label = 'artifact';

[DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
    savedir, instructions, assert_current_label);

%% print differences
disp(' ');
disp('You made these changes by clicking the GUI:');
disp('(Index: label_old --> label_new)');
for i=1:length(DATSTRUCT)
    if ~strcmp(DATSTRUCT(i).label_final, DATSTRUCT_MOD(i).label_final)
        disp([num2str(i) ': ' DATSTRUCT(i).label_final ' -> ' DATSTRUCT_MOD(i).label_final]);
    end
end

%% update label final int to match the string label.
DATSTRUCT_MOD = datstruct_mod_update_label_int(DATSTRUCT_MOD);

%% UPDATE DATSTRUCT
DATSTRUCT = DATSTRUCT_MOD;
clear DATSTRUCT_MOD

%% MERGE SUs... For each chan with multiple SUs, make a single figure;
% - DO THIS USING DATSTRUCT_FINAL, since aftre merging might want to run
% again. This only applies to SU.

% [DATSTRUCT_FINAL, LIST_MERGE_SU] = gui_waveforms_su_merge(DATSTRUCT_FINAL);
[~, LIST_MERGE_SU] = gui_waveforms_su_merge(DATSTRUCT);


%% Merge everything, including the SU
close all;

DATSTRUCT_MERGED = datstruct_merge(DATSTRUCT, [], LIST_MERGE_SU, ...
    indpeak, npre, npost, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR);

%% FINAL CLEANING OF MERGED DATA
%% Rerun double spike counter on DATSTRUCT_FINAL.

% 5. rerun spikes double count (really just for merged  MU since for SU is
% done above).
DRYRUN = false;
DATSTRUCT_MERGED = datstruct_remove_double_counted(DATSTRUCT_MERGED, indpeak, npre, npost, DRYRUN);

%% FINALLY, save final clean results.
SAVEDIR_FINAL_CLEAN = [SAVEDIR_FINAL '/CLEAN_AFTER_MERGE'];
mkdir(SAVEDIR_FINAL_CLEAN);

%% Recompute metrics after merging, useful for making figures
DOPLOT = false;
DATSTRUCT_MERGED = datstruct_compute_metrics(DATSTRUCT_MERGED, DOPLOT, ...
    indpeak, npre, npost);

%% Plots
% Before merging (but after correct labels)
savethis = [SAVEDIR_FINAL '/CLEAN_BEFORE_MERGE'];
mkdir(savethis);
datstruct_plot_summary(DATSTRUCT, savethis);
datstruct_plot_summary_premerge(DATSTRUCT, savethis);

% After merging
datstruct_plot_summary(DATSTRUCT_MERGED, SAVEDIR_FINAL_CLEAN);

%% #############################
%% DONE!

% Save DatStruct
% SAVE FINAL for this day
datstruct_save(DATSTRUCT_MERGED, SAVEDIR_FINAL, 'CLEAN_MERGED');
datstruct_save(DATSTRUCT, SAVEDIR_FINAL, 'CLEAN');
% save([SAVEDIR_FINAL '/DATSTRUCT_CLEAN_MERGED.mat'], 'DATSTRUCT_MERGED');
% save([SAVEDIR_FINAL '/DATSTRUCT_CLEAN.mat'], 'DATSTRUCT');

%% Save final waveforms
datstruct_plot_waveforms_all(DATSTRUCT_MERGED, SAVEDIR_FINAL_CLEAN, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR);
end