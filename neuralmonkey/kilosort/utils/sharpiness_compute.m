function sharpiness = sharpiness_compute(wf, PLOT)
%%! hacky score that should be high for things with sharp noise. sweeps over each qf. detects sudden changes in voltage.
% high change is scored as high sharpiness.
% PARAMS:
% - wf, (ntrials, ntimebins). NOTE: doesnt matter if wf are aligned across trials, since operates on individual wforms.
% RETURNS:
% - sharpiness, a single scalar score.

if ~exist('PLOT', 'var'); PLOT = false; end

list_s =[];
for j=1:size(wf, 1)
    wf1 = wf(j, :);
    wf1_diff = diff(wf1);
    %         wf1_diff = wf1_diff./mean(abs(wf1(1:end-1))); % normalizer by the magnitudes.
    %     wf1_diff = wf1_diff; % normalizer by the magnitudes.
    wf1_diff_diff = diff(wf1_diff); % look for sudden accel.
    wf1_diff_diff_abs = abs(wf1_diff_diff);
    
    if PLOT
        figure; hold on;
        % plot(wf');
        plot(wf1, '--k');
        
        figure; hold on;
        plot(wf1_diff);
        plot(wf1_diff_diff);
        plot(wf1_diff_diff_abs);
        assert(false);
    end
    
    % Collect vals over sliding windows.
    nprethis = 4; % n bins to compare this jump to (i.e,, if pre is flat, then jump.. that is sharp).
    npostthis = 2;
    windsize = nprethis + npostthis + 1;
    nwinds = size(wf1_diff_diff_abs,2)-windsize;
    vals = [];
    for k=nprethis+1:size(wf1_diff_diff_abs,2)-npostthis

        sdpre = mean(wf1_diff_diff_abs(k-nprethis:k-1));
        %             sdpost = mean(wf1_diff_diff_abs(k+1:k+npostthis));
        diffthis = wf1_diff_diff_abs(k);
        
        if sdpre==0
            % is usually because signal saturated. just use the entire trial as a fix.
            sdpre = mean(wf1_diff_diff_abs);
            % disp(wf1)
            % disp(wf1_diff)
            % disp(wf1_diff_diff)
            % disp(wf1_diff_diff_abs)
            % assert(false, 'fix this bug');
        end
        %             if sdpost==0
        %                 assert(false);
        %             end
        
        %         v = max((diffthis/sdpre), (diffthis/sdpost));
        v = (diffthis/sdpre);
        
        if isinf(v)
            assert(false, 'fix this bug');
        end
                
        vals(end+1) = v;
    end

    denom = median(vals);
    if denom==0
        denom = mean(vals);
        if denom==0
            denom = nan;
        end
    end

    s = max(vals)/denom; % penalize if there is sharpiness that s large, but over most windows its small.

    if isinf(s)
        disp(vals)
        disp(max(vals));
        disp(median(vals));
        assert(false, 'fix this bug');
    end

    list_s(end+1) = s;
    %     disp([sdpre diffthis sdpost]);
    %
    %     % max(wf1_diff_diff_abs)/median(wf1_diff_diff_abs)
    %     max(wf1_diff_diff_abs)/prctile(wf1_diff_diff_abs, [5])
end

sharpiness = mean(list_s);



% OLD VERSION:
% % CONCLUSION:" doesnt work that well. gives similar values for spikes and
% noise.

    list_sharpiness =[];
    for j=1:size(wf, 1)

        wf1 = wf(j, :);

    %     wf1_diff = diff(wf1)./wf1(1:end-1);
        wf1_diff = diff(wf1)./mean(abs(wf1(1:end-1))); % normalizer by the magnitudes.
        wf1_diff_diff = diff(wf1_diff);
        wf1_diff_diff_abs = abs(wf1_diff_diff);

        if false
            figure; hold on;
            % plot(wf');
            % plot(wf1, '--k');
            plot(wf1_diff);
            plot(wf1_diff_diff);
            plot(wf1_diff_diff_abs);
        end
        
%         s = max(wf1_diff_diff_abs)/(prctile(wf1_diff_diff_abs, [5]) + max(wf1_diff_diff_abs));
        s = max(wf1_diff_diff_abs)/median(wf1_diff_diff_abs);
        list_sharpiness(end+1) = s;
    end

    sharpy = mean(list_sharpiness);

end