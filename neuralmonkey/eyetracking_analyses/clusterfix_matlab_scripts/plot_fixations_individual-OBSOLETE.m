clear all; close all;

animal = 'Diego';
date = '230616';
session = '0';

base_path = '/home/kgg/Desktop/neuralmonkey/neuralmonkey/eyetracking/';
session_path = [base_path animal '-' date '-' session];

% load trialnums
load([session_path '/all_trialnums.mat']);

% store results in cell array
results = cell(1,length(trialnums));
eyedat = cell(1,length(trialnums));

assert(false, '@LT, avoid running this code, becuase this script is obsolete, since its missing the lines from plot_fixations_as_onexy.m that have @LT');

% load in eyedat, fs
for i = 1:length(trialnums)
    tnum = num2str(trialnums(i));
    disp(tnum);
    % load x,y,fs for this specific file
    load([session_path '/trial' tnum '.mat']);
    
    % convert xy to eyedat
    eyedat_trial = {[x; y]};
    % append xy to eyedat
    eyedat{i} = [x; y];
    
    % - get fs as # of samples per sec (is passed in as Hz)
    % - convert to double
    % - append to array
    fs = 1.0/double(fs);
    
    % run clusterfix
    results{i} = ClusterFix(eyedat_trial, fs);
    results{i} = results{i}{1}; % unpack to keep it simple
    
end

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
    