function kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label)
% After running finalize, you realize you mislabeld some clusters. Run this to upate the late.
% RUn for each set of labels to changes. 
% FINALIZE by running the finalize code again.
% PARAMS:
% - list_idxs_new, list of ints, indices that you want to change. these are labeled in the
% xaxis as "idx...". (Note these are not univeral indices. they are indices into the sliced dataset)
% e.g., list_idxs_new = [118 128 119];
% - change_kind, directory to find the plots, e.g.,, 'mua_noise'.
% - change_to_this_label, str, label to change to . e.g, 'mua'

%% MODIFY PARAMS
PATH_TO_SPIKES_CODE = '/gorilla1/code/spikes';
PATH_TO_NPY_CODE = '/gorilla1/code/npy-matlab';
PATH_TO_KILOSORT_CODE = '/gorilla1/code/kilosort-2.5';

[~, MACHINE] = system('hostname');
MACHINE = MACHINE(1:end-1); % weird character at end for gorilla.
switch MACHINE
    case 'lucast4-MS-7B98' % gorilla
        SAVEDIR_FINAL_SERVER =  '/mnt/Freiwald/kgupta/neural_data/postprocess'; % final, so all machines can access.
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
tmp1 = load([SAVEDIR_FINAL '/DATSTRUCT.mat']);
tmp2 = load([SAVEDIR_FINAL '/DATSTRUCT_CLEAN.mat']);
DATSTRUCT = tmp1.DATSTRUCT;
DATSTRUCT_CLEAN = tmp2.DATSTRUCT;
clear tmp1 tmp2

%% LOAD holder of manual changes

savepath = [SAVEDIR_FINAL_CLEAN '/MANUALCHANGES_TO_LABEL.mat'];
if exist(savepath, 'file')
    % Append changes
    tmp = load(savepath);
    MANUALCHANGES_TO_LABEL = tmp.MANUALCHANGES_TO_LABEL;
else
    % Initialize changes.
    MANUALCHANGES_TO_LABEL = {};
end

%% MAKE map from new to old indices

% First try loading pre-saved indices
paththis = [SAVEDIR_FINAL '/struct_inds_changed.mat'];
if exist(paththis, 'file')
    % only after 11/12/23
    tmp = load(paththis);
    struct_inds_changed = tmp.struct_inds_changed;
else
    % Recreat it . this is accurate, assuming that DASTRUT and DATSTRUCT_CLEAN are
    % up to date (i.e., qwhen the plots were made)
    struct_inds_changed = struct;

    for i=1:length(DATSTRUCT)
        if ~strcmp(DATSTRUCT(i).label_final, DATSTRUCT_CLEAN(i).label_final)

            % Collect indices
            f = [DATSTRUCT(i).label_final '_' DATSTRUCT_CLEAN(i).label_final];
            if isfield(struct_inds_changed, f)
                struct_inds_changed.(f) = [struct_inds_changed.(f), i];
            else
                struct_inds_changed.(f) = [i];
            end
        end
    end
end
map_indnew_to_indold = struct_inds_changed.(change_kind);

%% Given a list of indices (new) collect the original indices

SANITY = false;

if strcmp(change_kind, 'mua_noise')
    assert_label_old = 'noise';
else
    assert(false, 'code it');
end

% list_idxs_old = [];
for i=1:length(list_idxs_new)
    idx_new = list_idxs_new(i); % Index within the slice of DATSTRUCT that were of this change_kind
    idx_old = map_indnew_to_indold(idx_new); % INto DATSTRUCT and DATSTRUCT_CLEAN
    
    if SANITY
        datstruct_plot_waveforms_single(DATSTRUCT_CLEAN, idx_old, 100);
        DATSTRUCT_CLEAN(idx_old)
        disp('Check that the clkustnum is identical to int he suplbot:');
        DATSTRUCT_CLEAN(idx_old).clust
    end
    
%     list_idxs_old(end+1) = idx_old;
    
    % Chek that this index doesnt already exist
    if any(ismember(cellfun(@(x)x{1}, MANUALCHANGES_TO_LABEL), idx_old))
        disp(i);
        disp(idx_new);
        disp(idx_old);
        disp(cellfun(@(x)x{1}, MANUALCHANGES_TO_LABEL));
        displayCellArray(MANUALCHANGES_TO_LABEL);
        assert(false, 'only allowed one change');
    end 
    
    % APPEND TO CHANGES
    assert(strcmp(DATSTRUCT_CLEAN(idx_old).label_final, assert_label_old));
    
    disp(['old idx ' num2str(idx_old), ' will change from ' assert_label_old, ' to ', change_to_this_label]);

    MANUALCHANGES_TO_LABEL{end+1} = {idx_old, assert_label_old, change_to_this_label};
end

%% SAVE MANUALCHANGES_TO_LABEL
save(savepath, 'MANUALCHANGES_TO_LABEL');

disp('NOW RERUN kspostprocess_finalize_after_manual');



