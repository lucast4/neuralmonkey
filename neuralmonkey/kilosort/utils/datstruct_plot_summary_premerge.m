function datstruct_plot_summary_premerge(DATSTRUCT, SAVEDIR_FINAL)
%% Final plots, at level of clusters (i.e, not yet merged to level of channels).

if isempty(SAVEDIR_FINAL)
    VISIBLE = 'on';
else
    VISIBLE = 'off';
end

%% [FINAL PLOTS] Plot distributions over all clusts
% close all;

% % finally, plot distribution of scores
% figure;
% histogram(metric_results_arr);
% saveas(gcf, "scoredistributionhist.png")

list_isi = [DATSTRUCT.isi_violation_pct];
list_refract = [DATSTRUCT.Q];
list_snr = [DATSTRUCT.snr_final];

for j=1:2
    
    if j==1
        list_refract_this = list_isi;
        suff = 'isi_simple';
    elseif j==2
        list_refract_this = list_refract;
        suff = 'refractoriness_ks';
    else
        assert(false);
    end
    
    for k=1:2
        % Color by hand labels
        list_clust_group_name = {};
        for i=1:length(DATSTRUCT)
            if k==1
                % final auto labels
                list_clust_group_name{end+1} = DATSTRUCT(i).label_final;
                suff2 = 'final_label';
            elseif k==2
                % kilosort labels.
                list_clust_group_name{end+1} = DATSTRUCT(i).clust_group_name;
                suff2 = 'ks_label';
            else
                assert(false);
            end
        end
        
        % and metric vs. ISI
        FigH = figure('Position', get(0, 'Screensize'), 'visible', VISIBLE); hold on;
        xlabel(suff);
        ylabel('snr');
        title([suff2 ': red:good, b:mua, k=noise']);
        
        for i=1:length(list_refract_this)
            %     text(isi_violations_arr(i),metric_results_arr(i),num2str(inds_collected(i)));
            RSn = DATSTRUCT(i).RSn;
            batch = DATSTRUCT(i).batch;
            %     chan = DATSTRUCT(i).chan;
            clust = DATSTRUCT(i).clust;
            
            s = [num2str(RSn) '-' num2str(batch) '-' num2str(clust)];
            
            x = list_refract_this(i);
            y = list_snr(i);
            text(x,y,s, 'color', 'g', 'fontsize', 10)
            
        end
        
        inds = strcmp(list_clust_group_name, 'mua');
        plot(list_refract_this(inds), list_snr(inds), 'ob');
        
        inds = strcmp(list_clust_group_name, 'good') |  strcmp(list_clust_group_name, 'su');
        plot(list_refract_this(inds), list_snr(inds), 'or');
        
        inds = strcmp(list_clust_group_name, 'noise');
        plot(list_refract_this(inds), list_snr(inds), 'xk');
        
        inds = strcmp(list_clust_group_name, 'artifact');
        plot(list_refract_this(inds), list_snr(inds), 'om');
        
        fname = [SAVEDIR_FINAL '/scatter_allchans-snr-vs-' suff '-' suff2 '.png'];
        saveas(gcf, fname);
        
        xlim([0, 0.1]);
        fname = [SAVEDIR_FINAL '/scatter_allchans-snr-vs-' suff '-' suff2 '-ZOOM.png'];
        saveas(gcf, fname);
        close all;
    end
end