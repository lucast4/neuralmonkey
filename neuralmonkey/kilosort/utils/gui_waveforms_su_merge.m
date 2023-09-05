function [DATSTRUCT, LIST_MERGE_SU] = gui_waveforms_su_merge(DATSTRUCT, ...
    savepath_noext, SKIP_MANUAL_CURATION, SKIP_PLOTTING)
%% PARAMS
% - savepath_noext, leave empty to skip plotting.
%% For each chan with multiple SUs, make a single figure;
% - DO THIS USING DATSTRUCT_FINAL, since aftre merging might want to run
% again. This only applies to SU.

if ~exist('SKIP_PLOTTING', 'var'); SKIP_PLOTTING = false; end
if ~exist('SKIP_MANUAL_CURATION', 'var'); SKIP_MANUAL_CURATION = false; end

a = SKIP_MANUAL_CURATION==0 & SKIP_PLOTTING==0; % Generate plots and curate.
b = SKIP_MANUAL_CURATION==1 & SKIP_PLOTTING==0; % Generate figures, but no curation.
c = SKIP_MANUAL_CURATION==0 & SKIP_PLOTTING==1; % Load already saved figures, and do manual curation. (If figures don't exist, is OK, just continues).

assert(a | b | c);

MAKE_GUI = true;

%%

list_chan_global = 1:512;
% list_chan_global = unique([DATSTRUCT.chan_global]);
% INDICES_PLOT = [];
LIST_MERGE_SU = {}; % cell of 2-arays

for chan = list_chan_global
    
    if ~SKIP_PLOTTING
        % Then go straight to and loading.curation
        inds = [DATSTRUCT.chan_global]==chan & strcmp({DATSTRUCT.label_final}, 'su');
        good = sum(inds)>1;
        
        if good
            idxs = find(inds);
            
            nplots = length(idxs) + nchoosek(length(idxs), 2) + 1 + 1;
            ncols = ceil(sqrt(nplots));
            nrows = ncols;
            
            % plot
            disp([chan, DATSTRUCT(inds).index])
            
            pcols = lt_make_plot_colors(length(idxs));
            
            figure('Position', get(0, 'Screensize'), 'visible','on'); hold on; % full screen
            % figure; hold on;
            ct = 1;
            
            % 1) plot the waveforms
            YLIM = [-4500, 2000];
            for i=1:length(idxs)
                ind = idxs(i);
                
                subplot(nrows, ncols, ct); hold on;
                
                wf = DATSTRUCT(ind).waveforms;
                
                sharp = DATSTRUCT(ind).sharpiness;
                snr = DATSTRUCT(ind).snr_final;
                isi = DATSTRUCT(ind).isi_violation_pct;
                Q = DATSTRUCT(ind).Q;
                chan_global = DATSTRUCT(ind).chan_global;
                assert(chan==chan_global, 'sanityu check');
                clust = DATSTRUCT(ind).clust;
                label_final = DATSTRUCT(ind).label_final;
                
                if isfield(DATSTRUCT, 'index_before_merge')
                    idx = DATSTRUCT(ind).index_before_merge;
                else
                    idx = ind;
                end
                
                plot_waveforms_singleplot(wf, 100);
                % n = min(100, size(wf,1));
                % plot(wf(1:n,:)');
                ylabel(['isi' num2str(isi, '%0.2f') '-sh' num2str(sharp, '%0.1f')]);
                xlabel(['idx' num2str(idx)  '-cl' num2str(clust)]);
                if ~isempty(YLIM)
                    ylim(YLIM);
                end
                
                switch label_final
                    case 'noise'
                        pcol = 'k';
                    case 'mua'
                        pcol = 'b';
                    case 'su'
                        pcol = 'r';
                    case 'artifact'
                        pcol = 'm';
                    otherwise
                        disp(ind);
                        assert(false);
                end
                
                title(['chg' num2str(chan_global) '-snr' num2str(snr, '%0.2f') '-Q' num2str(Q, '%0.2f')], 'color', pcols{i});
                ct = ct+1;
                
            end
            
            % 2) Get cross-corr
            for i=1:length(idxs)
                for j=i+1:length(idxs)
                    
                    
                    i1 = idxs(i);
                    i2 = idxs(j);
                    s1 = DATSTRUCT(i1).times_sec_all;
                    s2 = DATSTRUCT(i2).times_sec_all;
                    
                    dt = 1/1000;
                    [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
                    Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
                    R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
                    
                    %             if Q<.2 && R<.05 % if both refractory criteria are met
                    %     if Q12<.25 && R<.05 % if both metrics are below threshold.
                    
                    % ----------------------
                    if true
                        subplot(nrows, ncols, ct); hold on;
                        title([num2str(i1) ' - ' num2str(i2)]);
                        
                        %                         t = 1:length(K);
                        t = 350:650;
                        K = K(t);
                        plot(K);
                        line([0 length(t)], [0 0]);
                        ylabel('K');
                        xlabel('lag (ms)');
                        ct = ct+1;
                        
                        % Plot the colors.
                        col1 = pcols{i};
                        col2 = pcols{j};
                        plot(0, 1, 'o', 'color', col1);
                        plot(1, 1, 'o', 'color', col2);
                    end
                    
                    if false
                        % NOt that useful, since it excludes timepoint 0
                        subplot(nrows, ncols, ct); hold on;
                        title([num2str(i1) ' - ' num2str(i2)]);
                        
                        plot(1:length(Qi), Qi/(max(Q00, Q01)));
                        line([1, length(Qi)], [0 0]);
                        ylabel('Q');
                        xlabel('lag (ms)');
                        ct = ct+1;
                        
                        % Plot the colors.
                        col1 = pcols{i};
                        col2 = pcols{j};
                        plot(0, 1, 'o', 'color', col1);
                        plot(1, 1, 'o', 'color', col2);
                    end
                    
                    % Chekc if there is double cotning of spikes.
                    
                    %
                    %                 ix = idxs(i);
                    %                 st = DATSTRUCT(ix).times_sec_all;
                    %                 [st_counts, t_edges]  = histcounts(st, linspace(min(st), max(st), 20));
                    %                 t_centers = t_edges(1:end-1) + diff(t_edges)/2;
                end
            end
            
            % 1) scatter plot of time stamps
            subplot(nrows, ncols, ct); hold on;
            ylabel('count (n spikes');
            xlabel('binned time');
            for i=1:length(idxs)
                col = pcols{i};
                ix = idxs(i);
                st = DATSTRUCT(ix).times_sec_all;
                [st_counts, t_edges]  = histcounts(st, linspace(min(st), max(st), 20));
                t_centers = t_edges(1:end-1) + diff(t_edges)/2;
                %            amps = DATSTRUCT(i).amps_wf;
                %            amps = i*ones(1, length(st));
                %            plot(st, amps, 'x');
                plot(t_centers, st_counts, '-o', 'color', col);
                text(t_centers(1), st_counts(1), num2str(ix), 'color', col, 'fontsize', 15);
                
            end
            ct = ct+1;
            YLIM = ylim;
            ylim([0, YLIM(2)]);
            
            subplot(nrows, ncols, ct); hold on;
            xlabel('idx1 - idx2 - concatted');
            ylabel('frac spks dbl cnted');
            %         xlabel('binned time');
            for i=1:length(idxs)
                for j=i+1:length(idxs)
                    
                    
                    i1 = idxs(i);
                    i2 = idxs(j);
                    s1 = DATSTRUCT(i1).times_sec_all;
                    s2 = DATSTRUCT(i2).times_sec_all;
                    
                    dt = 1/1000;
                    [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
                    Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
                    R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
                    
                    %             if Q<.2 && R<.05 % if both refractory criteria are met
                    %     if Q12<.25 && R<.05 % if both metrics are below threshold.
                    
                    % Double count rate
                    [~, ~, ~, ...
                        frac_spikes_double_counted_1] = datstruct_remove_double_counted_inner(s1);
                    
                    [~, ~, ~, ...
                        frac_spikes_double_counted_2] = datstruct_remove_double_counted_inner(s1);
                    
                    s12 = sort([s1; s2]);
                    [~, ~, ~, ...
                        frac_spikes_double_counted_12] = datstruct_remove_double_counted_inner(s12);
                    
                    plot([1 2 3], [frac_spikes_double_counted_1, ...
                        frac_spikes_double_counted_2, frac_spikes_double_counted_12], '-o', 'color', pcols{i});
                    plot([1], [frac_spikes_double_counted_1], '-o', 'color', pcols{i});
                    plot([2], [frac_spikes_double_counted_2], '-o', 'color', pcols{j});
                    
                    text(3.1, frac_spikes_double_counted_12, [num2str(i1) '-' num2str(i2)], 'fontsize', 12, 'color', pcols{j});
                    
                    ylim([0, 0.5]);
                    
                end
            end
            % Save it.
            if ~isempty(savepath_noext)
                disp(['Saving figure... ' savepath_noext '-' num2str(chan)]);
                if MAKE_GUI
                    % Then save with click functionalot
                    savefig(gcf, [savepath_noext '-' num2str(chan) '.fig']);
                end
                saveas(gcf, [savepath_noext '-' num2str(chan) '.png']);
                
                % save the indices
                paththis = [savepath_noext '-' num2str(chan) '-idxs.mat'];
                save(paththis, 'idxs');
            end
        end
    else
        % If skip plotting, then you must load previously saved plots. If
        % cant find, then continues.
        if isempty(savepath_noext)
            continue
        end
        paththis = [savepath_noext '-' num2str(chan) '.fig'];
        if exist(paththis, 'file')
            disp(paththis);
            openfig(paththis);
            
            % load the idsx
            paththis = [savepath_noext '-' num2str(chan) '-idxs.mat'];
            tmp = load(paththis);
            idxs = tmp.idxs;
        else
            disp('Skipping - could not find this file:');
            disp(paththis);
            continue % to next chan
        end
    end
    
    %% Ask user if want to merge
    if ~SKIP_MANUAL_CURATION
        merge = 'dummy';
        while ~isempty(merge)
            try
                merge = input('Want to merge any SU? Type them as array (e..g, [29 49], no quotes. If not, then ENTER');
                if isempty(merge)
                    disp('Empty...')
                    % then nothing to merge.
                else
                    if length(merge)==1
                        disp('TYPO? only 1 index ...')
                        disp(merge)
                    elseif length(unique(merge))==1
                        disp('TYPO? only 1 unique index. ...')
                        disp(merge)
                    else
                        % make sure you inputed indices that exist in the current fig.
                        ok = true;
                        for i=1:length(merge)
                            if ~ismember(merge(i), idxs)
                                disp('TYPO? index doesnt exist in figure')
                                ok = false;
                            end
                        end
                        if ok
                            disp(['... Merging these indices: ' num2str(merge)]);
                            LIST_MERGE_SU{end+1} = merge;
                        end
                    end
                end
            catch error
                disp('You typed this:');
                disp(merge);
                disp('CAUGHT ERROR in input (try again):');
                % disp(merge);
                merge = 'dummy';
            end
        end
    end
    
    close all;
    
end
end

