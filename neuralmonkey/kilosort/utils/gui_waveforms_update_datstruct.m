function DATSTRUCT = gui_waveforms_update_datstruct(DATSTRUCT, clickInfo, ...
	instructions, assert_current_label)
	
	if ~exist('assert_current_label', 'var'); assert_current_label = ''; end

    %% Interpret clickInfo and return modified DATSTRUCT
    % collect mods, in order so most recent click wins.
    indexmods = cell(1, length(DATSTRUCT));
    for i=1:length(clickInfo)
        idx = clickInfo{i}{1};
        but = clickInfo{i}{2}; % button
        disp([num2str(idx) ' -- ' but]);
        indexmods{idx} = instructions.(but); % interpret as string instruction
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
        	if ~isempty(assert_current_label)
	            if ~strcmp(label_old, assert_current_label)
	                disp(label_old);
	                disp(assert_current_label);
	                disp(ind)
	                assert(false);
	            end
	        end
            DATSTRUCT(ind).label_final = label_new;
            disp(['idx ' num2str(ind) ' - Changed from ' label_old ' to ' label_new]);
        end
    end
end