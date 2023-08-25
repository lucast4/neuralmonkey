function loadSavedFigureAndAddClickFunction()
    % Load the pre-saved figure
    fig = openfig('saved_figure.fig');
    
    % Get handles to the subplots
    subplots = findobj(fig, 'Type', 'axes');
    
    % Add click callbacks to subplots
    for i = 1:numel(subplots)
        set(subplots(i), 'ButtonDownFcn', {@subplotClickCallback, i});
        set(subplots(i), 'UIContextMenu', createContextMenus(i));
    end
end

function contextMenus = createContextMenus(subplotIndex)
    contextMenus = uicontextmenu;
    
    leftClickMenuItem = uimenu(contextMenus, 'Label', 'Left Click', 'Callback', {@contextMenuCallback, subplotIndex, 'left'});
    rightClickMenuItem = uimenu(contextMenus, 'Label', 'Right Click', 'Callback', {@contextMenuCallback, subplotIndex, 'right'});
end

function contextMenuCallback(~, ~, subplotIndex, clickType)
    disp(['Clicked on Subplot ' num2str(subplotIndex) ' with ' clickType ' click']);
    % Add your click action here
end

function subplotClickCallback(~, ~, subplotIndex)
    selectionType = get(gcf, 'SelectionType');
    if strcmp(selectionType, 'normal') % Left click
        contextMenuCallback([], [], subplotIndex, 'left');
    elseif strcmp(selectionType, 'alt') % Right click
        contextMenuCallback([], [], subplotIndex, 'right');
    end
end
