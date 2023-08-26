function datstruct_save(DATSTRUCT, savedir, suffix, also_save_each_unit)

if ~exist('also_save_each_unit', 'var'); also_save_each_unit = false; end

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

% Save each unit (item in DATSTRUCT), first ordered by global chan.
if also_save_each_unit
	disp('Saving separate4ly each unit...')

	% save each cluster (i.e., a unit)
	savedir = [savedir '/dat_individual'];
	mkdir(savedir)

	% 1) Sort by chan global, so that "unit" is in order
	T = struct2table(DATSTRUCT); % convert the struct array to a table
	sortedT = sortrows(T, 'chan_global'); % sort the table by 'DOB'
	DATSTRUCT_SORTED = table2struct(sortedT); % change it back to struct array if necessary

	for i=1:length(DATSTRUCT_SORTED)
	    dat = DATSTRUCT_SORTED(i);
	    path = [savedir '/unit_' num2str(i) '.mat'];
	    % chg = dat.chan_global;
	    % cl = dat.clust;
	%     path = [savedir '/unit_' num2str(i) chan_global_' num2str(chg) '-clust_' num2str(cl) '.mat'];
	    save(path, 'dat');
	    % disp(path)
	    % assert(false);
	    if mod(i, 20)==0
	    	disp(i)
	    end
	end
end

end