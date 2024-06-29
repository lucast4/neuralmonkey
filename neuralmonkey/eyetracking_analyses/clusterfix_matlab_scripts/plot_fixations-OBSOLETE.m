clear all; close all;

animal = 'Diego';
date = '230616';
session = '0';

base_path = '/home/kgg/Desktop/neuralmonkey/neuralmonkey/eyetracking/';
session_path = [base_path animal '-' date '-' session];

% load trialnums
load([session_path '/all_trialnums.mat']);

% create eyedat of fixed length; will contain xy for each trial
% do the same for fs
eyedat = cell(1,length(trialnums));
fs_arr = zeros(1,length(trialnums));

assert(false, '@LT, avoid running this code, becuase this script is obsolete, since its missing the lines from plot_fixations_as_onexy.m that have @LT');

% load in eyedat, fs
for i = 1:length(trialnums)
    tnum = num2str(trialnums(i));
    
    % load x,y,fs for this specific file
    load([session_path '/trial' tnum '.mat']);
    
    % append xy to eyedat
    eyedat{i} = [x; y];
    
    % - get fs as # of samples per sec (is passed in as Hz)
    % - convert to double
    % - append to array
    fs_arr(i) = 1.0/double(fs);
end

% make sure all fs are the same
if fs_arr ~= fs_arr(1)
    disp('error: sampling rate differs across trials');
    return;
end

% % do in batches to help k-mean clustering along (was failing)
% results = cell(1,length(trialnums));
% batch_size = 20;

% run ClusterFix
results = ClusterFix(eyedat, fs_arr(1));

% plot results for each trial
for i = 1:length(trialnums)
    tnum = num2str(trialnums(i));
    disp(tnum)
    figure('Name',tnum);
    
    % plot xy
    x = eyedat{i}(1,:);
    y = eyedat{i}(2,:);
    plot(x,y)
    hold on

    % get fixation times and plot each, alternating colors to separate
    fixationtimes = results{i}.fixationtimes;

    for j=1:length(fixationtimes)
        %disp(fixationtimes(1,i))
        time_window = fixationtimes(1,j):fixationtimes(2,j);
        if mod(j,2)==0
            plot(x(time_window), y(time_window), 'r');
        else
            plot(x(time_window), y(time_window), 'g');
        end
        hold on
    end
    
    % done with this plot
    hold off
end
    