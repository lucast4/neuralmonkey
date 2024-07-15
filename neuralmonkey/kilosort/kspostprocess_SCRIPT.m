
clear all; close all; 
ANIMAL = 'Pancho';
DATE = 240619;
% SKIP_RAW_PLOTS_EACH_CLUST = true;
SKIP_LOADING_DATSTRUCT=true;

% Run these, no without human intervention
%kspostprocess_extract(ANIMAL, DATE);
%kspostprocess_metrics_and_label(ANIMAL, DATE, SKIP_RAW_PLOTS_EACH_CLUST);
    
% This is the human manual curation step.
kspostprocess_manual_curate_merge(ANIMAL, DATE, SKIP_LOADING_DATSTRUCT);

%%
% After curating, run this to finalize saved data
kspostprocess_finalize_after_manual(ANIMAL, DATE)


% Done!

%% To manually change a label after finalized
clear all; close all;

ANIMAL = 'Pancho';
%DATE = 220715;
DATES = [220719 220609 220608 220606 220717 ];
IDXS = {[17 32 51 63 12 23 26]
[19 12 14 33 35 39 50 53 56 59 60 61 65 71 79 80 81 83 85 88 94 97 104 105 112 114 120]
[16 18 24 25 31 33 55 64 60 71 91]    
[12 37 34 39 42 45 46 54 72 86 89 94]
[1 2 3 37 28 31 43 48 50 56 78 79 64 71 74 75 76 77 82 83 92 94 102 105 106 111 113 114 122 138 141]
};

%%% Manually enter the indices to change
% These are the indices in the subplotse xlabel, e.g., idx69..
% '/mnt/Freiwald_kgupta/kgupta/neural_data/postprocess/final_clusters/Pancho/220715/CLEAN_BEFORE_MERGE/curated_changes_waveforms/mua_noise/waveform_gridplots_byfinallabel/noise-sorted_by_changlobal-4.png

%list_idxs_new = [69 70];
change_kind = 'mua_noise'; % the original change to want to undo.
change_to_this_label = 'mua'; % what you want to change the label to.
%kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);

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
%kspostprocess_finalize_after_manual(ANIMAL, DATE)

for i=1:length(DATES)
    DATE = DATES(i);
    list_idxs_new = IDXS{i};
    
    kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);
    kspostprocess_finalize_after_manual(ANIMAL, DATE)
end

%%
clear all; close all;

ANIMAL = 'Diego';
DATES = [230616 230924 230915 230616];
IDXS = {
[1 5 14 17 18 19 21 23 50 29 33 37 38 39 53 41 42 55 57 58 59 60 61 62 63 67 68 70 73 78 79 81 84 87 91 94 96 102 103 106 107 108 120 116 123 126 128 131 132 133 134 138 145 142 148 150 151 161 165 167 172 173 175 176 179 180 182 183 184 185 190 191 192]    
[197 198 210 204 206 1 4 12 14 19 20 22 23 24 25 39 50 51 30 36 33 42 38 43 45 46 48 52 56 57 83 73 84 76 77 79 90 95 98 101 102 106 108 113 114 115 119 120 128 129 130 131 132 143 144 152 156 157 158 160 161 164 165 167 168 170 171 174 207 202 177 178 179 181 186 187 190 192 195 196 197 210 204]
[1 4 5 18 3 9 11 12 13 14 16 21 27 28 29 30 31 32 33 34 35 36 37 39 40 48 42 43 44 50 53 59 61 72 79 80 82 83 85 92 93 99 100 103 106 107 110 111 116 120 122 124 128 134 131 132 139 140 147 148 149 150 151 152 157 159 161 165 167 168 174 175 176 179 181 182]
[1 5 14 17 19 21 23 50 29 33 37 192 38 53 41 55 57 58 59 60 61 62 196 63 73 79 81 84 91 94 100 102 190 193 106 107 120 113 116 123 125 126 128 131 132 133 134 138 145 142 150 161 165 189 167 172 173 179 180 194 182 183 186 187 184 185 189 190 191 192 163 164 165 166 188 167 170 171 172 173 174 175 176 178 182]
};

change_kind = 'mua_noise';
change_to_this_label = 'mua';

for i=1:length(DATES)
    DATE = DATES(i);
    list_idxs_new = IDXS{i};
    kspostprocess_changelabel_after_finalize(ANIMAL, DATE, list_idxs_new, change_kind, change_to_this_label);
    kspostprocess_finalize_after_manual(ANIMAL, DATE)
end
