function datstruct_plot_chan_su_vs_mu(DATSTRUCT, savedir)

list_chan_global = 1:512;

for chan = list_chan_global
    % Then go straight to and loading.curation
    inds_su = find([DATSTRUCT.chan_global]==chan & strcmp({DATSTRUCT.label_final}, 'su'));
    inds_mu = find([DATSTRUCT.chan_global]==chan & strcmp({DATSTRUCT.label_final}, 'mua'));
    
    good = length(inds_su)>0 & length(inds_mu)>0;
    
    if good
        %             idxs = find(inds);
        %
        %             nplots = length(idxs) + nchoosek(length(idxs), 2) + 1 + 1;
        %             ncols = ceil(sqrt(nplots));
        %             nrows = ncols;
        
        % plot
        %             disp([chan, DATSTRUCT(inds).index])
        %
        %             pcols = lt_make_plot_colors(length(idxs));
        %
        %             figure('Position', get(0, 'Screensize'), 'visible','on'); hold on; % full screen
        %             % figure; hold on;
        %             ct = 1;
        %
        %             % 1) plot the waveforms
        %             YLIM = [-4500, 2000];
        
        figure('Position', get(0, 'Screensize'), 'visible','off'); hold on; % full screen
        nplots = length(inds_su)*length(inds_mu) + length(inds_su) + length(inds_mu);
        ncols = ceil(sqrt(nplots));
        if ncols<4
            ncols = 4;
        end
        nrows = ceil(nplots/ncols);
        ct = 1;
        
        % 1) plot the waveforms
        YLIM = [-4500, 2000];
        idxs = [inds_su inds_mu];
        pcols = lt_make_plot_colors(length(idxs));
        
        for i=1:length(idxs)
            ind = idxs(i);
            
            subplot(nrows, ncols, ct); hold on;
            
            wf = DATSTRUCT(ind).waveforms;
            st = DATSTRUCT(ind).times_sec_all;
            
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
            ylabel(['isi' num2str(isi, '%0.2f') '-sh' num2str(sharp, '%0.1f')]);
            xlabel(['chg' num2str(chan_global)  '-cl' num2str(clust) '[n=' num2str(length(st)) ']']);
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
            
            % title(['chg' num2str(chan_global) '-snr' num2str(snr, '%0.2f') '-Q' num2str(Q, '%0.2f')], 'color', pcols{i});
            title(['idx' num2str(ind) '-idxb4merge' num2str(idx) '-snr' num2str(snr, '%0.2f') '-Q' num2str(Q, '%0.2f')], 'color', pcol);
            ct = ct+1;
        end
        
        for i=1:length(inds_su)
            for j=1:length(inds_mu)
                
                i1 = inds_su(i);
                i2 = inds_mu(j);
                s1 = DATSTRUCT(i1).times_sec_all;
                s2 = DATSTRUCT(i2).times_sec_all;
                
                dt = 1/1000;
                [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
                Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
                R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
                
                % ----------------------
                subplot(nrows, ncols, ct); hold on;
                title([num2str(i1) ' - ' num2str(i2)]);
                
                col1 = pcols{i};
                col2 = pcols{j};
                
                %                         t = 1:length(K);
                t = 350:650;
                K = K(t);
                plot(K);
                line([0 length(t)], [0 0]);
                ylabel('K', 'color', col1);
                xlabel('lag (ms)', 'color', col2);
                
                % Plot the colors.
                %                 plot(0, 1, 'o', 'color', col1);
                %                 plot(1, 1, 'o', 'color', col2);
                ct = ct+1;
                
            end
        end
        path = [savedir '/su_vs_mu-chan_' num2str(chan) '.png'];
        disp(['Saving ...' path]);
        saveas(gcf, path);
    end
end
end