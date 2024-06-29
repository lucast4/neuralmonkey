function get_clusterfix_results_mat(animal, date, session, base_path, do_plots)
    if ~exist('do_plots', 'var'); do_plots = false; end
    session_path = [base_path '/' animal '-' date '-' session];
    xy_mats_path = [session_path '/raw_xy_mats'];

    % load trialnums
    load([xy_mats_path '/all_ntrialnums.mat']);
    load([xy_mats_path '/all_trialcodes.mat']);
    trialcodes = cellstr(trialcodes);

    %% store x,y,time data as one continuous for all trials
    x_all = [];
    y_all = [];
    times_all = [];
    %bounding_inds_all = [];

    % store all sampling rates (fs)
    fs_all = zeros(1,length(neuraltrialnums));

    % store start, end indices of each trial (to index into x_all, y_all, time_all)
    ntrial_start_end_inds = zeros(3,length(neuraltrialnums));

    % load in eyedat, fs for each trial, and concatenate them
    for i = 1:length(neuraltrialnums)
        ntnum = neuraltrialnums(i);
        % load x,y,fs for this specific file
        load([xy_mats_path '/ntrial' num2str(ntnum) '.mat']);

        % store tnum
        ntrial_start_end_inds(1,i) = ntnum;
        % store start index
        ntrial_start_end_inds(2,i) = length(x_all)+1;
        % append xy to eyedat
        x_all = [x_all x];
        y_all = [y_all y];
        times_all = [times_all times_xy];
        % store end index
        ntrial_start_end_inds(3,i) = length(x_all);

        %bounding_inds_all = [bounding_inds_all bounding_inds];

        % - fs_hz is passed in as Hz
        % - convert to number of samples per second (integer, but make double)
        % - append to array
        fs_all(i) = 1.0/double(fs_hz);
    end

    % make sure all fs are the same
    if any(fs_all ~= fs_all(1)) % @LT, added any().
        disp('error: sampling rate differs across trials');
        return;
    end

    %% using concatenated array, run ClusterFix
    eyedat = {[x_all; y_all]};
    tic
    results = ClusterFix(eyedat, fs_all(1));
    disp('--- ENTIRE clusterfix took this long:');
    toc

    %% save ClusterFix results into .mat files
    RESULTS = struct;

    % repartition into individual trials and add to RESULTS
    for i = 1:length(neuraltrialnums)
        % get ntrialnum, and start/end indices
        ntnum = ntrial_start_end_inds(1,i);
        tcode = trialcodes{i};
        nt_start_ind = ntrial_start_end_inds(2,i);
        nt_end_ind = ntrial_start_end_inds(3,i);

        % get x,y,times for just this trial
        x_nt = x_all(nt_start_ind:nt_end_ind);
        y_nt = y_all(nt_start_ind:nt_end_ind);
        times_nt = times_all(nt_start_ind:nt_end_ind);

        % get fixation times, mean positions, within this trial
        fixation_times_all = results{1}.fixationtimes; % 2xN array with start,end times
        fixation_centroids_all = results{1}.fixations;
        nt_fixation_inds = [];
        nt_fixation_centroids = [];

        for j=1:length(fixation_times_all)
            fixation_start_ind = fixation_times_all(1,j);
            fixation_end_ind = fixation_times_all(2,j);

            % check that fixation belongs to this trial
            if (fixation_start_ind >= nt_start_ind) && (fixation_end_ind <= nt_end_ind)

                % get raw indices
                nt_fixation_start_ind = fixation_start_ind-nt_start_ind+1;
                nt_fixation_end_ind = fixation_end_ind-nt_start_ind;
                nt_fixation_inds = [nt_fixation_inds [nt_fixation_start_ind; nt_fixation_end_ind]];
                nt_fixation_centroids = [nt_fixation_centroids [fixation_centroids_all(1,j); fixation_centroids_all(2,j)]];
            end
        end

        % get saccade times within this trial
        saccade_times_all = results{1}.saccadetimes;
        nt_saccade_inds = [];

        for j=1:length(saccade_times_all)
            saccade_start_ind = saccade_times_all(1,j);
            saccade_end_ind = saccade_times_all(2,j);

            % check that fixation belongs to this trial
            if (saccade_start_ind >= nt_start_ind) && (saccade_end_ind <= nt_end_ind)

                % get raw times from indices
                nt_saccade_start_ind = saccade_start_ind-nt_start_ind+1;
                nt_saccade_end_ind = saccade_end_ind-nt_start_ind;
                nt_saccade_inds = [nt_saccade_inds [nt_saccade_start_ind; nt_saccade_end_ind]];
            end
        end

        RESULTS(i).neuraltrialnum = ntnum;
        RESULTS(i).trialcode = tcode;
        RESULTS(i).fs = fs_all(i);
        RESULTS(i).x = x_nt;
        RESULTS(i).y = y_nt;
        RESULTS(i).times = times_nt;
        RESULTS(i).fixation_inds = nt_fixation_inds;
        RESULTS(i).fixation_centroids = nt_fixation_centroids;
        RESULTS(i).saccade_inds = nt_saccade_inds;
    end
    % RESULTS.clusterfix_results = results;
    % RESULTS.eyedat = eyedat;
    % RESULTS.fs_arr = fs_all;
    % RESULTS.start_ends = ntrial_start_end_inds;
    % RESULTS.times_all = times_all;
    %RESULTS.bounding_inds_all = bounding_inds_all;
    % RESULTS.neuraltnums = neuraltrialnums;
    % RESULTS.tcodes = trialcodes;
    disp('saving clusterfix_results.mat ...');
    save([session_path '/clusterfix_results.mat'], 'RESULTS', '-v7');
    disp('finished saving clusterfix_results.mat');

    %% plot results for each trial
    if do_plots
        savedir_figs = [session_path '/trial_plots']; % @LT, saving, too many to plot.
        mkdir(savedir_figs);
        for i = 1:length(RESULTS)
            trial_results = RESULTS(i);
            ntnum = trial_results.neuraltrialnum;
            disp(ntnum)
            %t_start_ind = ntrial_start_end_inds(2,i);
            %t_end_ind = ntrial_start_end_inds(3,i);
            figure('Name', num2str(ntnum), 'visible', 'off'); % @LT, annoying to pop up.

            % plot xy
            x = trial_results.x;
            y = trial_results.y;
            plot(x,y)
            hold on

            % get fixation times and plot each, alternating colors to separate
            fixation_times_all = trial_results.fixation_inds;

            for j=1:length(fixation_times_all)
                fix_inds = fixation_times_all(1,j):fixation_times_all(2,j);
                %disp(fixationtimes(1,i))
                %time_window = fixation_times_all(1,j):fixation_times_all(2,j);

                % check that fixation belongs to this trial
                %if (time_window(1) >= t_start_ind) && (time_window(end) <= t_end_ind)
            if mod(j,2)==0
                plot(x(fix_inds), y(fix_inds), 'r');
            else
                plot(x(fix_inds), y(fix_inds), 'g');
            end
            hold on
                %end
                %plot(x(fix_inds), y(fix_inds), 'g');
            end

            % done with this plot
            hold off

            % Save plot and close, @LT
            saveas(gcf, [savedir_figs '/trial_' num2str(ntnum) '_fixations.png']); 
            close all;
        end
    end
end
