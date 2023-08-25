function datstruct_save(DATSTRUCT, savedir, suffix)

if length(suffix)>0
    name = ['DATSTRUCT_' suffix];
else
    name = 'DATSTRUCT';
end

% 1) save .mat
save([savedir '/' name '.mat'], 'DATSTRUCT', '-v7.3');

% 2) save as table.
T = struct2table(DATSTRUCT); % convert the struct array to a table
sortedT = sortrows(T, 'chan_global'); % sort the table by 'DOB'
cols_remove = {'times_sec_all', 'waveforms','amps_wf', 'times_sec_all_BEFORE_REMOVE_DOUBLE'}; % time series...
for i=1:length(cols_remove)
    col = cols_remove{i};
    if isfield(DATSTRUCT, col)
        sortedT = removevars(sortedT, {col});
    end
end
% DATSTRUCT_SORTED = table2struct(sortedT); % change it back to struct array if necessary
writetable(sortedT, [savedir '/' name '-table.csv']);
end