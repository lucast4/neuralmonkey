clear all; close all; 
ANIMAL = 'Pancho';
DATE = 230623;

%%
SKIP_LOADING_DATSTRUCT = true;
% This is the human manual curation step.
kspostprocess_manual_curate_merge(ANIMAL, DATE, SKIP_LOADING_DATSTRUCT);

%% after curating, now load datstruct and finalize.
%clear all; close all


clear all; close all; 
ANIMAL = 'Pancho';
% DATE = 220719;
% DATES = [221015 221024 221220 230616 230622 230623];

%DATES = [220719  220918  221024  221218  230103  230105  230613  230622, ...
%    220724  221015  221217  221220  230104  230612  230616  230623]; % all excpt 221015

DATES = [220915 230125 230126 230127]
for i=1:length(DATES)
    DATE = DATES(i);
    
%     try
    % Run these, no without human intervention
    kspostprocess_finalize_after_manual(ANIMAL, DATE)
%     catch err
%     end
end

