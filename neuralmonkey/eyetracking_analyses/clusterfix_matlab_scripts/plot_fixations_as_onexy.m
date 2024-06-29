clear all; close all;

animal = 'Diego';
date = '230626';
session = '0-fixation-middle';
% session = '00'; % for testing just on trials 1:20

base_path = '/home/kgg/Desktop/neuralmonkey/neuralmonkey/eyetracking/';
session_path = [base_path animal '-' date '-' session];

% load trialnums
load([session_path '/all_ntrialnums.mat']);
load([session_path '/all_trialcodes.mat']);
%trialnums = [136];

%% store xy data as one continuous for all trials
x_all = [];
y_all = [];
times_all = [];
bounding_inds_all = [];

% create fs of fixed length
fs_arr = zeros(1,length(neuraltrialnums));
% store start, end indices of each trial
start_ends = zeros(3,length(neuraltrialnums));

% load in eyedat, fs
for i = 1:length(neuraltrialnums)
    tnum = neuraltrialnums(i);
    disp(tnum);
    % load x,y,fs for this specific file
    load([session_path '/ntrial' num2str(tnum) '.mat']);
    
    % store tnum
    start_ends(1,i) = tnum;
    % store start index
    start_ends(2,i) = length(x_all)+1;
    % append xy to eyedat
    x_all = [x_all x];
    y_all = [y_all y];
    times_all = [times_all times];
    bounding_inds_all = [bounding_inds_all bounding_inds];
    % store end index
    start_ends(3,i) = length(x_all);

    % - get fs as # of samples per sec (is passed in as Hz)
    % - convert to double
    % - append to array
    fs_arr(i) = 1.0/double(fs_hz);
end

time_inds = 1:length(x_all);

% make sure all fs are the same
if any(fs_arr ~= fs_arr(1)) % @LT, added any().
    disp('error: sampling rate differs across trials');
    return;
end

% % do in batches to help k-mean clustering along (was failing)
% results = cell(1,length(trialnums));
% batch_size = 20;

%% run ClusterFix
eyedat = {[x_all; y_all]};
tic
results = ClusterFix(eyedat, fs_arr(1));
disp('--- ENTIRE clusterfix took this long:');
toc

%% @LT Save results
RESULTS = struct;
RESULTS.clusterfix_results = results;
RESULTS.eyedat = eyedat;
RESULTS.fs_arr = fs_arr;
RESULTS.start_ends = start_ends;
RESULTS.times_all = times_all;
RESULTS.bounding_inds_all = bounding_inds_all;
RESULTS.neuraltnums = neuraltrialnums;
RESULTS.tcodes = trialcodes;
save([session_path '/clusterfix_results.mat'], 'RESULTS', '-v7');

%% plot results for each trial
savedir_figs = [session_path '/trial_plots']; % @LT, saving, too many to plot.
mkdir(savedir_figs);
for i = 1:length(neuraltrialnums)
    tnum = num2str(neuraltrialnums(i));
    disp(tnum)
    t_start_ind = start_ends(2,i);
    t_end_ind = start_ends(3,i);
    figure('Name',tnum, 'visible', 'off'); % @LT, annoying to pop up.

    % plot xy
    x = x_all(t_start_ind:t_end_ind);
    y = y_all(t_start_ind:t_end_ind);
    plot(x,y)
    hold on

    % get fixation times and plot each, alternating colors to separate
    fixationtimes = results{1}.fixationtimes;

    for j=1:length(fixationtimes)
        %disp(fixationtimes(1,i))
        time_window = fixationtimes(1,j):fixationtimes(2,j);
        
        % check that fixation belongs to this trial
        if (time_window(1) >= t_start_ind) && (time_window(end) <= t_end_ind)
            if mod(j,2)==0
                plot(x_all(time_window), y_all(time_window), 'r');
            else
                plot(x_all(time_window), y_all(time_window), 'g');
            end
            hold on
        end
    end

    % done with this plot
    hold off
    
    % Save plot and close, @LT
    saveas(gcf, [savedir_figs '/trial_' num2str(tnum) '_fixations.png']); 
    close all;
end

%% plot x position vs. time
for i = 1:length(neuraltrialnums)
    tnum = num2str(neuraltrialnums(i));
    disp(tnum)
    t_start_ind = start_ends(2,i);
    t_end_ind = start_ends(3,i);
    figure('Name', tnum, 'visible', 'off'); % @LT, annoying to pop up.

    % plot x-time
    x = x_all(t_start_ind:t_end_ind);
    %y = y_all(t_start_ind:t_end_ind);
    t = time_inds(t_start_ind:t_end_ind);
    plot(t,x)
    hold on

    % get fixation times and plot each, alternating colors to separate
    fixationtimes = results{1}.fixationtimes;

    for j=1:length(fixationtimes)
        %disp(fixationtimes(1,i))
        time_window = fixationtimes(1,j):fixationtimes(2,j);
        
        % check that fixation belongs to this trial
        if (time_window(1) >= t_start_ind) && (time_window(end) <= t_end_ind)
            if mod(j,2)==0
                plot(time_inds(time_window), x_all(time_window), 'r');
            else
                plot(time_inds(time_window), x_all(time_window), 'g');
            end
            hold on
        end
    end

    % done with this plot
    hold off
    
    % Save plot and close, @LT
    saveas(gcf, [savedir_figs '/trial_' num2str(tnum) '_x-vs-time.png']); 
    close all;
end

%% plot y position vs. time
for i = 1:length(neuraltrialnums)
    tnum = num2str(neuraltrialnums(i));
    disp(tnum)
    t_start_ind = start_ends(2,i);
    t_end_ind = start_ends(3,i);
    figure('Name', tnum, 'visible', 'off'); % @LT, annoying to pop up.

    % plot y-time
    %x = x_all(t_start_ind:t_end_ind);
    y = y_all(t_start_ind:t_end_ind);
    t = time_inds(t_start_ind:t_end_ind);
    plot(t,y)
    hold on

    % get fixation times and plot each, alternating colors to separate
    fixationtimes = results{1}.fixationtimes;

    for j=1:length(fixationtimes)
        %disp(fixationtimes(1,i))
        time_window = fixationtimes(1,j):fixationtimes(2,j);
        
        % check that fixation belongs to this trial
        if (time_window(1) >= t_start_ind) && (time_window(end) <= t_end_ind)
            if mod(j,2)==0
                plot(time_inds(time_window), y_all(time_window), 'r');
            else
                plot(time_inds(time_window), y_all(time_window), 'g');
            end
            hold on
        end
    end

    % done with this plot
    hold off
    
    % Save plot and close, @LT
    saveas(gcf, [savedir_figs '/trial_' num2str(tnum) '_y-vs-time.png']); 
    close all;
end
