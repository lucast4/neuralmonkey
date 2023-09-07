clear all; close all; 
ANIMAL = 'Pancho';
DATE = 220719;
SKIP_LOADING_DATSTRUCT = true;
% This is the human manual curation step.
kspostprocess_manual_curate_merge(ANIMAL, DATE, SKIP_LOADING_DATSTRUCT);

%% after curating, now load datstruct and finalize.
clear all; close all
kspostprocess_finalize_after_manual('Pancho', '230623')
