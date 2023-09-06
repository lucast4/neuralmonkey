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
