%% get waveforms that are shifted so that the peaks align.
function [wf_shifted, new_peak_ind] = get_shifted_wf(wf_input, ispositive, orig_peak_ind, npre, npost)
    % Shift each waveform in time to align them. does this by finding peak for each wform (within a window
    % centered at peak) then shifting so this max is at same index for all dat
    % - wf_input, (ndat, ntimebins)
    % RETURNS:
    % - wf_shifted
    % - new_peak_ind, index into wf_shifted(:, new_peak_ind), that is peak time.

    % if 10, then new peak is at 11
    if ~exist('npre', 'var'); npre = 12; end
    if ~exist('npost', 'var'); npost = 18; end

    % if positive spike, temporarily invert
    if ispositive
        wf_input = -wf_input;
    end
    
    % sometimes spikes are misaligned, so get the correct spike peak time
    % [~, inds_min_times] = min(wf_input, [], 2);
    WIND = 4; % will look back and forw from peak time by this much. this ensures doesnt shift to noisy peak.
    [~, inds_min_times] = min(wf_input(:, (orig_peak_ind-WIND):(orig_peak_ind+WIND)), [], 2);
    inds_min_times = inds_min_times + orig_peak_ind - WIND - 1;


    % further subsampling, to pick out just the spike
    new_peak_ind = npre+1;
    wf_shifted = [];
    
%     disp(inds_min_times);
    for i=1:size(wf_input,1)
        indthis = inds_min_times(i);
        
        % if spike is noise, just keep as is, otherwise will throw error
        if (indthis-npre <= 0) || (indthis+npost > size(wf_input,2)) % i.e. min value is near beginning or end, so probably noise
%             disp(indthis);200
            indthis = orig_peak_ind; % don't shift, keep as is
        end
        
        wfthis = wf_input(i, indthis-npre:indthis+npost);
        % 
        wf_shifted = [wf_shifted; wfthis];
    end
  
    % if positive spike, invert again to get original wf
    if ispositive
        wf_shifted = -wf_shifted;
    end
    
end