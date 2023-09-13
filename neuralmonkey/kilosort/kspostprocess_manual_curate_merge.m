function kspostprocess_manual_curate_merge(ANIMAL, DATE, SKIP_LOADING_DATSTRUCT)
% given data that has been kilosorted, and extracred using quickscript_spikes_extract
% run this to curate and autoamticlaly label data.

if ~exist('SKIP_LOADING_DATSTRUCT', 'var'); SKIP_LOADING_DATSTRUCT = false; end
assert(SKIP_LOADING_DATSTRUCT==true, 'too much coded now assuming this...');

%% MODIFY PARAMS

PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';

% LOADDIR_BASE = '/lemur2/kilosort_data'; % location of kilosorted data
% SAVEDIR_LOCAL = '/lemur2/kilosort_temp'; % fast ssd
[~, MACHINE] = system('hostname');
MACHINE = MACHINE(1:end-1); % weird character at end for gorilla.
switch MACHINE
    case 'lucast4-MS-7B98' % gorilla
        SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald_kgupta/kgupta/neural_data/postprocess'; % final, so all machines can access.
        %         LOADDIR_BASE = '/mnt/Freiwald_kgupta/kgupta/neural_data'; % location of kilosorted data
        %         SAVEDIR_LOCAL = '/gorilla4/neural_preprocess_kilosort'; % fast ssd
    case 'lemur'
        SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/kgupta/neural_data/postprocess'; % final, so all machines can access.
        %         LOADDIR_BASE = '/lemur2/kilosort_data'; % location of kilosorted data
        %         SAVEDIR_LOCAL = '/lemur2/kilosort_temp'; % fast ssd
    case 'LAPTOP-5ROGVGP5' % rig laptop
        SAVEDIR_FINAL_SERVER =  'y:/emmy_data01/kgupta/neural_data/postprocess'; % final, so all machines can access.
        %         LOADDIR_BASE = '/lemur2/kilosort_data'; % location of kilosorted data
        %         SAVEDIR_LOCAL = '/lemur2/kilosort_temp'; % fast ssd
    otherwise
        disp(['MACHINE: ' MACHINE])
        disp([MACHINE(1)])
        disp([MACHINE(end)])
        assert(false,'add it here');
        %         SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/kgupta/neural_data/postprocess'; % final, so all machines can access.
end

% LOADDIR_BASE = '/mnt/Freiwald/kgupta/neural_data'; % location of kilosorted data
% SAVEDIR_LOCAL = '/gorilla4/neural_preprocess_kilosort'; % fast ssd
% SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/ltian/neural_data/preprocessing/kilosort_postprocess'; % final, so all machines can access.

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

% SAVEDIR_BASE = SAVEDIR_LOCAL;
% SAVEDIR_BASE_DATE = [SAVEDIR_BASE '/' ANIMAL '/' num2str(DATE)];

SAVEDIR_FINAL_BASE = SAVEDIR_FINAL_SERVER;
SAVEDIR_FINAL = [SAVEDIR_FINAL_BASE '/final_clusters/' ANIMAL '/' num2str(DATE)];

%% LOAD GLOBAL PARAMS

[indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR] = quickscript_config_load();


%% Chek that prev step done

path =[SAVEDIR_FINAL '/DONE_kspostprocess_metrics_and_label.mat'];
assert(exist(path, 'file')==2, ['Did not complete prev step, ' path]);

%% OUTPUT OF CLEAN
SAVEDIR_FINAL_CLEAN = [SAVEDIR_FINAL '/CLEAN_AFTER_MERGE'];
mkdir(SAVEDIR_FINAL_CLEAN);


%% Load datstruct, with metrics already computed (ther ouptut of quickscript_spikes)

if SKIP_LOADING_DATSTRUCT
    DATSTRUCT = [];
else
    tmp = load([SAVEDIR_FINAL '/DATSTRUCT.mat']);
    DATSTRUCT = tmp.DATSTRUCT;
    clear tmp
end

%% [LOAD GUI] and manually curate


savedir = [SAVEDIR_FINAL '/waveform_gridplots_GUI'];

DATSTRUCT_MOD = DATSTRUCT;
STRUCT_CLICKINFO = []; % To collect clickInfos

% STEPS

%%% (1) noise, add those that are MU
figpath = 'noise-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'cancel');
assert_current_label = 'noise';
[DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN);

% paththis = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
% % - check if already done
% if exist(paththis, 'file')
%     SKIP = input(['Found saved clickInfo for ' figpath '. ** Type y to redo. anything else to skip'], 's');
% else
%     SKIP = '';
% end
% if ~strcmp(SKIP, 'y')
%     [DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
%         savedir, instructions, assert_current_label);
%     STRUCT_CLICKINFO = [STRUCT_CLICKINFO struct('figpath', figpath, 'clickInfo', {clickInfo})];
%     % - interim save
%     save(paththis, 'clickInfo');
%     disp(['** Saved clickInfo to: ' paththis]);
% end

%%% (2) MU, remove those that are noise
figpath = 'MU-sorted_by_snr';
instructions = struct('left', 'to_noise', 'right', 'to_su');
assert_current_label = 'mua';
[DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN);

% 2. SU, remove that that are MU.
figpath = 'SU-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'to_artifact');
assert_current_label = 'su';
[DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN);

% 3. artifact, keep those that are SU.
figpath = 'artifact-sorted_by_sharpiness';
instructions = struct('left', 'to_su', 'right', 'to_mua');
assert_current_label = 'artifact';
[DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN);

% [OPTIONALLY] run this if you want top remove all MU
figpath = 'MU-ALL';
instructions = struct('left', 'to_noise', 'right', 'to_su');
assert_current_label = 'mua';
[DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN);

%% FINALLY, save final clean results.
paththis = [SAVEDIR_FINAL_CLEAN '/STRUCT_CLICKINFO.mat'];
save(paththis, 'STRUCT_CLICKINFO');
disp('** Saved STRUCT_CLICKINFO to:');
disp(paththis);


%% If you previously saved SU_merge gui figs, then load them.

SKIP_MANUAL_CURATION = false;
SKIP_PLOTTING = true;
savepath_noext = [savedir '/SUmerge'];
[~, LIST_MERGE_SU, LIST_NOTES] = gui_waveforms_su_merge(DATSTRUCT, savepath_noext, ...
    SKIP_MANUAL_CURATION, SKIP_PLOTTING);
save([SAVEDIR_FINAL_CLEAN '/LIST_MERGE_SU.mat'], 'LIST_MERGE_SU');
notefile = [SAVEDIR_FINAL_CLEAN '/MERGE_SU_LIST_NOTES.txt'];
for i=1:length(LIST_NOTES)
    s = LIST_NOTES{i};
    writematrix(s,notefile,'WriteMode','append');
end


%% print differences
if ~SKIP_LOADING_DATSTRUCT
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
    SKIP_MANUAL_CURATION = false;
    savepath_noext = [];
    [~, LIST_MERGE_SU, LIST_NOTES] = gui_waveforms_su_merge(DATSTRUCT, [], SKIP_MANUAL_CURATION);
    save([SAVEDIR_FINAL_CLEAN '/LIST_MERGE_SU.mat'], 'LIST_MERGE_SU');
    notefile = [SAVEDIR_FINAL_CLEAN '/MERGE_SU_LIST_NOTES.txt'];
    for i=1:length(LIST_NOTES)
        s = LIST_NOTES{i};
        writematrix(s,notefile,'WriteMode','append');
    end
    
    %% Merge everything, including the SU
    close all;
    DATSTRUCT_MERGED = datstruct_merge(DATSTRUCT, [], LIST_MERGE_SU, ...
        indpeak, npre, npost, THRESH_SU_SNR, ...
        THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
        THRESH_ARTIFACT_ISI, MIN_SNR);
    
    %% FINAL CLEANING OF MERGED DATA
    %% Rerun double spike counter on DATSTRUCT_FINAL.
    
    % 5. rerun spikes double count (really just for merged MU vs, each other, and
    % SU vs. merged MU. For SU vs. SU pairs, already done above).
    DRYRUN = false;
    DATSTRUCT_MERGED = datstruct_remove_double_counted(DATSTRUCT_MERGED, indpeak, npre, npost, DRYRUN);
    
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
    plot_decision_boundaries = false;
    datstruct_plot_waveforms_all(DATSTRUCT_MERGED, SAVEDIR_FINAL_CLEAN, THRESH_SU_SNR, ...
        THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
        THRESH_ARTIFACT_ISI, MIN_SNR, plot_decision_boundaries);
end
end

function [DATSTRUCT_MOD, STRUCT_CLICKINFO] = load_and_curate_single(DATSTRUCT_MOD, STRUCT_CLICKINFO,  ...
    figpath, instructions, assert_current_label, ...
    SAVEDIR_FINAL, SAVEDIR_FINAL_CLEAN)

savedir = [SAVEDIR_FINAL '/waveform_gridplots_GUI'];
paththis = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
% - check if already done
disp(['!! You are doing: ' figpath]);
if exist(paththis, 'file')
    %     disp(1);
    DO = input(['Found saved clickInfo for ' figpath '. ** Type y to redo. anything else to skip'], 's');
else
    %     disp(2);
    DO = input('Type n to skip...', 's');
    if ~strcmp(DO, 'n')
        DO = 'y';
    end
end
if strcmp(DO, 'y')
    [DATSTRUCT_MOD, clickInfo] = gui_waveforms_load_figs(DATSTRUCT_MOD, figpath, ...
        savedir, instructions, assert_current_label);
    STRUCT_CLICKINFO = [STRUCT_CLICKINFO struct('figpath', figpath, 'clickInfo', {clickInfo})];
    % - interim save
    save(paththis, 'clickInfo');
    disp(['** Saved clickInfo to: ' paththis]);
end
end
