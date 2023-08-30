clear all; close all; 
ANIMAL = 'Pancho';
%DATE = 221015;
% DATES = [221015 221024 221220 230616 230622 230623];
DATES = [220719 220724 220918 221024 221217 221220 230616 230622 230623]; % all excpt 221015

parfor i=1:length(DATES)
    DATE = DATES(i);
       
    % Run these, no without human intervention
    kspostprocess_extract(ANIMAL, DATE);
    kspostprocess_metrics_and_label(ANIMAL, DATE);
end

% This is the human manual curation step.
%kspostprocess_manual_curate_merge(ANIMAL, DATE);

% Done!
