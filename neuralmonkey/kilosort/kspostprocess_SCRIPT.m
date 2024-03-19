clear all; close all; 
ANIMAL = 'Pancho';
DATE = 220609;
SKIP_RAW_PLOTS_EACH_CLUST = true;
SKIP_LOADING_DATSTRUCT=true;

% Run these, no without human intervention
%kspostprocess_extract(ANIMAL, DATE);
%kspostprocess_metrics_and_label(ANIMAL, DATE, SKIP_RAW_PLOTS_EACH_CLUST);
    
% This is the human manual curation step.
kspostprocess_manual_curate_merge(ANIMAL, DATE, SKIP_LOADING_DATSTRUCT);

% After curating, run this to finalize saved data
kspostprocess_finalize_after_manual(ANIMAL, DATE)


% Done!

%% To manually change a label after finalized
clear all; close all;

ANIMAL = 'Diego';
DATE = 230616;

%%% Manually enter the indices to change
% These are the indices in the subplotse xlabel, e.g., idx69..
% '/mnt/Freiwald_kgupta/kgupta/neural_data/postprocess/final_clusters/Pancho/220715/CLEAN_BEFORE_MERGE/curated_changes_waveforms/mua_noise/waveform_gridplots_byfinallabel/noise-sorted_by_changlobal-4.png

list_idxs_new = [2 7 8 20 9 13 15 16 49 26 28 51 31 52 34 43 45 46 47 48 65 66 72 89 90 82 83 119 99 105 121 115 117 118 127 130 155 158 160 163 164 166 170 171 174 178 186 187 188];
change_kind = 'mua_noise'; % the original change to want to undo.
change_to_this_label = 'mua'; % what you want to change the label to.
kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);

% YOu can run this multiple times, each time it will append new changes
%list_idxs_new = [80 91];
%change_kind = 'mua_noise'; % the original change to want to undo.
%change_to_this_label = 'mua'; % what you want to change the label to.
%kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);

%%% Finally, rerun the finalize script to redo all finalizing, but this
%%% time incorproating the chagnes above.
% See the code starting with "APPLYING MANUAL CHANGES"
kspostprocess_finalize_after_manual(ANIMAL, DATE)

%%%
DATE = 230915;
list_idxs_new = [47 41 54 62 60 74 94 98 163 114 126 133 138 146 153 162 172 173];
kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);
kspostprocess_finalize_after_manual(ANIMAL, DATE)

DATE = 230924;
list_idxs_new = [6 11 16 17 18 40 28 31 41 78 91 92 99 112 137 138 139 159 163 175 180 182 193 194 208 206];
kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);
kspostprocess_finalize_after_manual(ANIMAL, DATE)