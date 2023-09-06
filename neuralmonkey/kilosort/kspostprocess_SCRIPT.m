clear all; close all; 
ANIMAL = 'Pancho';
DATE = 230623;

% Run these, no without human intervention
% kspostprocess_extract(ANIMAL, DATE);
kspostprocess_metrics_and_label(ANIMAL, DATE);
    
% This is the human manual curation step.
%kspostprocess_manual_curate_merge(ANIMAL, DATE);

% Done!
