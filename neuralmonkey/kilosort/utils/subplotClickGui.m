function multiFigureSubplotClickGUI()
    numFigures = 2;  % Number of figures
    numSubplots = 4; % Number of subplots in each figure
    
    for f = 1:numFigures
        % Create a new figure for each iteration
        fig = figure('Name', ['Figure ' num2str(f)], 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
        
        % Create subplots
        subplots = zeros(1, numSubplots);
        
        for i = 1:numSubplots
            subplots(i) = subplot(2, 2, i, 'Parent', fig);
            plot(subplots(i), rand(10, 1));
            title(subplots(i), sprintf('Subplot %d', i));
            set(subplots(i), 'ButtonDownFcn', {@subplotClickCallback, f, i});
        end
    end
end

function subplotClickCallback(~, event, figureIndex, subplotIndex)
    selectionType = get(gcf, 'SelectionType');
    if strcmp(selectionType, 'normal') % Left click
        disp(['Left clicked on Figure ' num2str(figureIndex) ', Subplot ' num2str(subplotIndex)]);
        % Add your left-click action here
    elseif strcmp(selectionType, 'alt') % Right click
        disp(['Right clicked on Figure ' num2str(figureIndex) ', Subplot ' num2str(subplotIndex)]);
        % Add your right-click action here
    end
end