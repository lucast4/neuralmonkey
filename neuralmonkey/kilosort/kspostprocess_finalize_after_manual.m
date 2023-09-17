function kspostprocess_finalize_after_manual(ANIMAL, DATE)
% given data that has been kilosorted, and extracred using quickscript_spikes_extract
% run this to curate and autoamticlaly label data.

%% MODIFY PARAMS

PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';

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
    case 'ltbonobo'
        SAVEDIR_FINAL_SERVER = '/home/kgg/mnt/Freiwald/kgupta/neural_data/postprocess';
    otherwise
        disp(['MACHINE: ' MACHINE])
        disp([MACHINE(1)])
        disp([MACHINE(end)])
        assert(false,'add it here');
        %         SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/kgupta/neural_data/postprocess'; % final, so all machines can access.
end

addpath(genpath(PATH_TO_SPIKES_CODE));
addpath(genpath(PATH_TO_NPY_CODE));
addpath(genpath([PATH_TO_KILOSORT_CODE '/postProcess']));

SAVEDIR_FINAL_BASE = SAVEDIR_FINAL_SERVER;
SAVEDIR_FINAL = [SAVEDIR_FINAL_BASE '/final_clusters/' ANIMAL '/' num2str(DATE)];

%% LOAD GLOBAL PARAMS

[indpeak, wind_spike, npre, npost, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR] = quickscript_config_load();

%% Chek that prev step done

% No need, since prev step makes files.
% path =[SAVEDIR_FINAL '/DONE_kspostprocess_metrics_and_label.mat'];
% assert(exist(path, 'file')==2, ['Did not complete prev step, ' path]);

%% OUTPUT OF CLEAN
SAVEDIR_FINAL_CLEAN = [SAVEDIR_FINAL '/CLEAN_AFTER_MERGE'];

%% Load datstruct, with metrics already computed (ther ouptut of quickscript_spikes)
disp('Loading DATSTRUCT, may take a while....');
tmp = load([SAVEDIR_FINAL '/DATSTRUCT.mat']);
DATSTRUCT = tmp.DATSTRUCT;
clear tmp

%% [LOAD GUI] and manually curate

% savedir = [SAVEDIR_FINAL '/waveform_gridplots_GUI'];

DATSTRUCT_MOD = DATSTRUCT;

% STEPS
%%% (1) noise, add those that are MU
figpath = 'noise-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'cancel');
assert_current_label = 'noise';

path = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
load(path, 'clickInfo');
DATSTRUCT_MOD = gui_waveforms_update_datstruct(DATSTRUCT_MOD, clickInfo, ...
    instructions, assert_current_label);

%%% (1) noise, add those that are MU
figpath = 'MU-sorted_by_snr';
instructions = struct('left', 'to_noise', 'right', 'to_su');
assert_current_label = 'mua';

path = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
load(path, 'clickInfo');
DATSTRUCT_MOD = gui_waveforms_update_datstruct(DATSTRUCT_MOD, clickInfo, ...
    instructions, assert_current_label);


% 2. SU, remove that that are MU.
figpath = 'SU-sorted_by_snr';
instructions = struct('left', 'to_mua', 'right', 'to_artifact');
assert_current_label = 'su';

path = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
load(path, 'clickInfo');
DATSTRUCT_MOD = gui_waveforms_update_datstruct(DATSTRUCT_MOD, clickInfo, ...
    instructions, assert_current_label);


% 3. artifact, keep those that are SU.
figpath = 'artifact-sorted_by_sharpiness';
instructions = struct('left', 'to_su', 'right', 'to_mua');
assert_current_label = 'artifact';

path = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
load(path, 'clickInfo');
DATSTRUCT_MOD = gui_waveforms_update_datstruct(DATSTRUCT_MOD, clickInfo, ...
    instructions, assert_current_label);


% [OPTIONALLY] run this if you want top remove all MU
figpath = 'MU-ALL';
instructions = struct('left', 'to_noise', 'right', 'to_su');
assert_current_label = 'mua';

path = [SAVEDIR_FINAL_CLEAN '/clickInfo-' figpath '.mat'];
load(path, 'clickInfo');
DATSTRUCT_MOD = gui_waveforms_update_datstruct(DATSTRUCT_MOD, clickInfo, ...
    instructions, assert_current_label);

%% update label final int to match the string label.
DATSTRUCT_MOD = datstruct_mod_update_label_int(DATSTRUCT_MOD);

%% print differences
% save changes to text file
notefile = [SAVEDIR_FINAL_CLEAN '/final_cleaning_changes.txt'];

disp(' ');
disp('You made these changes by clicking the GUI:');
disp('(Index: label_old --> label_new)');
writematrix('(Index: label_old --> label_new)',notefile,'WriteMode','append');
for i=1:length(DATSTRUCT)
    if ~strcmp(DATSTRUCT(i).label_final, DATSTRUCT_MOD(i).label_final)
        s = [num2str(i) ': ' DATSTRUCT(i).label_final ' -> ' DATSTRUCT_MOD(i).label_final];
        disp(s);
        writematrix(s,notefile,'WriteMode','append');
    end
end

%% If you previously saved SU_merge gui figs, then load them.
load([SAVEDIR_FINAL_CLEAN '/LIST_MERGE_SU.mat'], 'LIST_MERGE_SU');

%% Deal with situations where SU merge conflicts with mods above.
% Some of the SU already converted to MU. You have mixture of SU and MU.
% You could merge averything into a MU. Option 1: throw out the MU that was
% previoulsty SU. Option 2: more conservative, convert everything to MU and
% merge them.

LIST_MERGE_SU_GOOD = {};
for i=1:length(LIST_MERGE_SU)
    merge_su = LIST_MERGE_SU{i};
        
    labels = {DATSTRUCT_MOD(merge_su).label_final};
    
    if all(strcmp(labels, 'su'))
        % good, all still SU
        LIST_MERGE_SU_GOOD{end+1} = merge_su;
        continue
    else
        % then convert all the SU to MU.
        s = ['[Converting su to mu] This merge_su is not all su... ' num2str(merge_su)];
        writematrix(s,notefile,'WriteMode','append');
        for j=1:length(merge_su)
            idx = merge_su(j);
            % if this is SU, convert to MU
            if strcmp(DATSTRUCT_MOD(idx).label_final, 'su')
                DATSTRUCT_MOD(idx).label_final = 'mua';
                
                s = [num2str(idx) ': ' 'su' ' -> ' 'mua'];
                disp(s);
                writematrix(s,notefile,'WriteMode','append');
            end
        end
        % And exclude it from merges
    end    
end

% All of the SU already converted to MU or artifact. Then skip this SU
% merge.

disp('Old LIST_MERGE_SU:');
disp(LIST_MERGE_SU);

disp('New LIST_MERGE_SU:');
disp(LIST_MERGE_SU_GOOD);

LIST_MERGE_SU = LIST_MERGE_SU_GOOD;

%% UPDATE DATSTRUCT
% rename
DATSTRUCT = DATSTRUCT_MOD;
clear DATSTRUCT_MOD

%% MERGE SU.

%% MERGE SUs... For each chan with multiple SUs, make a single figure;
% - DO THIS USING DATSTRUCT_FINAL, since aftre merging might want to run
% again. This only applies to SU.

%% Merge everything, including the SU
close all;
[DATSTRUCT_MERGED, LIST_MERGE_SU_FINAL_LABEL] = datstruct_merge(DATSTRUCT, [], LIST_MERGE_SU, ...
    indpeak, npre, npost, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR);

%%

s = '======= SU MERGES PERFORMED:';
writematrix(s,notefile,'WriteMode','append');

for i=1:length(LIST_MERGE_SU_FINAL_LABEL)
    merge_su = LIST_MERGE_SU_FINAL_LABEL{i}{1};
    label_final = LIST_MERGE_SU_FINAL_LABEL{i}{2};
    
    s = [num2str(merge_su) ' merged, with final label: ' label_final];
    disp(s);
    writematrix(s,notefile,'WriteMode','append');
end


%% TODO: Merge SU-MU
% ACTUALLY: no need. this is actually done since there is step to convert
% SU to MU. Then this MU will autoamtically be merged with other MU.
% Process identical to above in datstruct_merge. Except always final
% classification should be MU. 

%% FINAL CLEANING OF MERGED DATA
%% Rerun double spike counter on DATSTRUCT_FINAL.

% 5. rerun spikes double count (really just for merged MU vs, each other, and
% SU vs. merged MU. For SU vs. SU pairs, already done above).
DRYRUN = false;
DATSTRUCT_MERGED = datstruct_remove_double_counted(DATSTRUCT_MERGED, indpeak, npre, npost, DRYRUN);

%% Recompute metrics after merging, useful for making figures
% BUT DONT reclassify, this defeatts purpose of manual curation.
% For merged, you have already reclassified SU-SU merges in
% datstruct_merge. Assumes that SU-MU and MU-MU merges always result in 
% MU.
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


%% a figure for each chan - compare su and mu.
savedir = [SAVEDIR_FINAL_CLEAN '/actually_merged_su_vs_mu'];
mkdir(savedir);
datstruct_plot_chan_su_vs_mu(DATSTRUCT_MERGED, savedir);

%% Save final waveforms
plot_decision_boundaries = false;
datstruct_plot_waveforms_all(DATSTRUCT_MERGED, SAVEDIR_FINAL_CLEAN, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR, plot_decision_boundaries);

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
