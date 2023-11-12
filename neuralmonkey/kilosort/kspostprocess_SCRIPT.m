clear all; close all; 
ANIMAL = 'Pancho';
DATE = 230623;
SKIP_RAW_PLOTS_EACH_CLUST = true;

% Run these, no without human intervention
% kspostprocess_extract(ANIMAL, DATE);
kspostprocess_metrics_and_label(ANIMAL, DATE, SKIP_RAW_PLOTS_EACH_CLUST);
    
% This is the human manual curation step.
%kspostprocess_manual_curate_merge(ANIMAL, DATE);

% Done!

%% To manually change a label after finalized
clear all; close all;

ANIMAL = 'Pancho';
DATE = 220715;

%%% Manually enter the indices to change
% These are the indices in the subplotse xlabel, e.g., idx69..
% '/mnt/Freiwald_kgupta/kgupta/neural_data/postprocess/final_clusters/Pancho/220715/CLEAN_BEFORE_MERGE/curated_changes_waveforms/mua_noise/waveform_gridplots_byfinallabel/noise-sorted_by_changlobal-4.png

list_idxs_new = [69 70];
change_kind = 'mua_noise'; % the original change to want to undo.
change_to_this_label = 'mua'; % what you want to change the label to.
kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);

% YOu can run this multiple times, each time it will append new changes
list_idxs_new = [80 91];
change_kind = 'mua_noise'; % the original change to want to undo.
change_to_this_label = 'mua'; % what you want to change the label to.
kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);

%%% Finally, rerun the finalize script to redo all finalizing, but this
%%% time incorproating the chagnes above.
% See the code starting with "APPLYING MANUAL CHANGES"
kspostprocess_finalize_after_manual(ANIMAL, DATE)