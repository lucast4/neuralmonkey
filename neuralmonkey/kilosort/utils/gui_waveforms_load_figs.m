function [DATSTRUCT, clickInfo] = gui_waveforms_load_figs(DATSTRUCT, figpath, savedir, ...
        instructions, assert_current_label);
    % Load guis.
    % EG:
        % figpath = 'MU-sorted_by_snr';
        % instructions = struct('left', 'to_noise', 'right', 'cancel');
        % assert_current_label = 'mua';    
    % RETURNS:
    % - clickInfo, cell array of info for each click. NOTE: this will also be int he base wqorkspace.
    % RUN: - clean storage

    % Warn user starting a new curation.
    figure; hold on;
    text(1,1, 'DONE! starting new curation. ENTER to continue...')
    xlim([0.8, 8])
    ylim([0.5 1.5])
    input('')
    close all;

    clickInfo = {};
    assignin('base', 'clickInfo', clickInfo);

    % - print instructions
    F = fieldnames(instructions);
    disp('-----------------------');
    disp(['- Showing these plots: ' figpath]);
    disp('- INSTRUCTIONS. Click on subplots to do the following:');
    for i=1:length(F)
        disp(['****' F{i} ' click: ' instructions.(F{i})]);
    end
    disp('- (most recent click overrides all prev)');
    disp('------------------------');

    % - open each figure
    path_noext = [savedir '/' figpath '*.fig'];
    list_figs = dir(path_noext);
    for i=1:length(list_figs)
        path = [list_figs(i).folder '/' list_figs(i).name];
        disp(['Opening this figure: '  list_figs(i).name]);
        openfig(path);
        key = [];
        while isempty(key)
            disp('Type "x" and press ENTER to undo previous (subplot color will remain)')
            key = input('Type nothing and press ENTER to close figure and open next...', 's');

            if isempty(key)
                close all; 
                key = 'done';
                % break % out of while
            elseif strcmp(key, 'x')
                clickInfo = evalin('base', 'clickInfo');
                if length(clickInfo)>0
                    disp('Undid the last click: ')
                    disp([num2str(clickInfo{end}{1}) ' -- ' num2str(clickInfo{end}{2})]);
                    clickInfo = clickInfo(1:end-1);
                else
                    disp('... clickInfo is empty. not undoing anything.')
                end
                key = [];
                assignin('base', 'clickInfo', clickInfo);
            end
        end
        close all;
    end

    % Return the clickInfo
    clickInfo = evalin('base', 'clickInfo');

    %% Interpret clickInfo and return modified DATSTRUCT
    % collect mods, in order so most recent click wins.
    indexmods = cell(1, length(DATSTRUCT));
    for i=1:length(clickInfo)
        idx = clickInfo{i}{1};
        but = clickInfo{i}{2};
        disp([num2str(idx) ' -- ' but]);
        indexmods{idx} = instructions.(but);
    end

    % Update DATSTRUCT
    for ind = 1:length(indexmods)
        if isempty(indexmods{ind})
            continue
        end
        % sanity check that you are modifiying it from the correct starting label.
        label_old = DATSTRUCT(ind).label_final;
        switch indexmods{ind}
            case 'to_noise'
                % make this a noise cluster
                label_new = 'noise';
                do_change = true;
            case 'to_mua'
                label_new = 'mua';
                do_change = true;
            case 'to_su'
                label_new = 'su';
                do_change = true;
            case 'to_artifact'
                label_new = 'artifact';
                do_change = true;
            case 'cancel'
                % do nothing. 
                do_change = false;
            otherwise
                assert(false);
        end
        if do_change
            if ~strcmp(label_old, assert_current_label)
                disp(label_old);
                disp(assert_current_label);
                disp(ind)
                assert(false);
            end
            DATSTRUCT(ind).label_final = label_new;
            disp(['idx ' num2str(ind) ' - Changed from ' label_old ' to ' label_new]);
        end
    end

end