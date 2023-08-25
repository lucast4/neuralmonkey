function map_figsubplot_to_index = plot_waveforms_sorted_by(DATSTRUCT, values_sort, ...
    value_range, savepath_noext, ...
    exclude_labels, close_plots, MAKE_GUI, YLIM, INDICES_PLOT, ROWS, COLS)
%% Gridplot, example waveforms, ordered by some variable.
% PARAMS:
% - values_sort, array of scalars, matchign DATSTRUCT, which variable to sort subplots by
% - value_range, [min, max], will only plot in this range.
% - title_vars, cell array of str, to label each sbiplot

if ~exist('exclude_labels', 'var'); exclude_labels = {}; end
if ~exist('close_plots', 'var'); close_plots = true; end
if ~exist('MAKE_GUI', 'var'); MAKE_GUI = false; end
if ~exist('YLIM', 'var'); YLIM = []; end
if ~exist('INDICES_PLOT', 'var'); INDICES_PLOT = []; end
if ~exist('ROWS', 'var'); ROWS = 3; end
if ~exist('COLS', 'var'); COLS = 7; end

% if MAKE_GUI
%     close_plots=false;
% end

NPLOT = 80;

if close_plots
    INVISIBLE = true;
else
    INVISIBLE = false;
end

figcount=1;
subplotrows=ROWS;
subplotcols=COLS;
fignums_alreadyused=[];
hfigs=[];
hsplots = [];

[~, indsort] = sort(values_sort);

if isempty(value_range)
    value_range = [min(values_sort)-1 max(values_sort)+1];
end

nplot = sum(values_sort>value_range(1) & values_sort<value_range(2));
nfig = nplot/(subplotrows * subplotcols);
disp(['Making this many subplots: ', num2str(nplot), ', ' num2str(nfig) ' figs.']);

map_figsubplot_to_index = nan(ceil(nfig), (subplotrows * subplotcols));

if MAKE_GUI
    % make variables to save click information
    clickInfo = {};
    assignin('base', 'clickInfo', clickInfo);
end

for i=1:length(indsort)
    ind = indsort(i);
    label_final = DATSTRUCT(ind).label_final;
    
    if ~isempty(INDICES_PLOT)
        if ~ismember(ind, INDICES_PLOT)
            continue
        end
    end
    if ismember(label_final, exclude_labels)
        continue
    end
    
    wf = DATSTRUCT(ind).waveforms;

    val = values_sort(ind);
    if val<value_range(1) | val>value_range(2)
        continue
    end
    
    % % skip if is noise
    % if strcmp(DATSTRUCT(ind).label_final, 'noise')
    %     continue
    % end
    
    sharp = DATSTRUCT(ind).sharpiness;
    snr = DATSTRUCT(ind).snr_final;
    isi = DATSTRUCT(ind).isi_violation_pct;
    Q = DATSTRUCT(ind).Q;
    chan_global = DATSTRUCT(ind).chan_global;
    clust = DATSTRUCT(ind).clust;
    
    
    % Make plot
    [fignums_alreadyused, hfigs, figcount, hsplot, fignum, ...
        subplot_num]=lt_plot_MultSubplotsFigs('', subplotrows, ...
        subplotcols, fignums_alreadyused, hfigs, figcount, ...
        INVISIBLE);

    plot_waveforms_singleplot(wf, NPLOT);
    
    ylabel(['isi' num2str(isi, '%0.2f') '-sh' num2str(sharp, '%0.1f')]);
    xlabel(['idx' num2str(ind)  '-cl' num2str(clust)]);
    if ~isempty(YLIM)
        ylim(YLIM);
    end
    
    switch label_final
        case 'noise'
            pcol = 'k';
        case 'mua'
            pcol = 'b';
        case 'su'
            pcol = 'r';
        case 'artifact'
            pcol = 'm';
        otherwise
            disp(ind);
            assert(false);
    end
    
    title(['chg' num2str(chan_global) '-snr' num2str(snr, '%0.2f') '-Q' num2str(Q, '%0.2f')], 'color', pcol);
    
    % save map
    map_figsubplot_to_index(fignum, subplot_num) = ind;
    if MAKE_GUI
        set(hsplot, 'ButtonDownFcn', {@subplotClickCallback, fignum, subplot_num, ind});
    end
end
for i=1:length(hfigs)
    disp(['Saving figure... ' savepath_noext '-' num2str(i)]);
    if MAKE_GUI
        % Then save with click functionalot
        savefig(hfigs(i), [savepath_noext '-' num2str(i) '.fig']);
    end
    saveas(hfigs(i), [savepath_noext '-' num2str(i) '.png']);
end

if close_plots
    close all;
end
end


function subplotClickCallback(src, event, figureIndex, subplotIndex, datIndex)
selectionType = get(gcf, 'SelectionType');

% Get the click information from the base workspace
clickInfo = evalin('base', 'clickInfo');
% extendType = event.Extend;

if strcmp(selectionType, 'normal') % Left click
    disp(['Left clicked on Figure ' num2str(figureIndex) ', Subplot ' num2str(subplotIndex)]);
    % Add your left-click action here
    clickInfo{end+1} = {datIndex, 'left'};
    set(src, 'Color', [0.8 0.8 0.8]); % Light gray color
elseif strcmp(selectionType, 'alt') % Right click
    disp(['Right clicked on Figure ' num2str(figureIndex) ', Subplot ' num2str(subplotIndex)]);
    % Add your right-click action here
    clickInfo{end+1} = {datIndex, 'right'};
    set(src, 'Color', [0.9 0.5 0.5]); % Light red color
    % elseif strcmp(extendType, 'on') % Middle click
    %     set(src, 'Color', [0.5 0.9 0.5]); % Light green color
end
disp(['... data index:', num2str(datIndex)]);
assignin('base', 'clickInfo', clickInfo);
end
