function datstruct_plot_summary(DATSTRUCT, SAVEDIR_FINAL)
%% [FINAL PRINT SUMMARY]

if ~exist('SAVEDIR_FINAL', 'var'); SAVEDIR_FINAL= []; end

if isempty(SAVEDIR_FINAL)
    VISIBLE = 'on';
else
    VISIBLE = 'off';
end

%% for each unique rs-batch-chan, count it
FigH = figure('Position', get(0, 'Screensize'), 'visible', VISIBLE); hold on;
xlabel('chan_global');
ylabel('-1 noclust, 0 noise, 1 mu, 2 su');
title('final clust labels');

list_chans_global = 1:512;
list_labels = {};
% list_n_independent_clusters = [];
for cg = list_chans_global
    inds = [DATSTRUCT.chan_global]==cg;
    labels = [DATSTRUCT(inds).label_final_int];
    if false
        disp([cg labels]);
    end
    col = [rand rand rand]*0.7 + 0.3;
    for j=1:length(labels)
        jitter = 0.2*([0 rand]-0.5);
        plot(cg+jitter(1), labels(j)+jitter(2), 'o', 'color', col);
    end
    if length(labels)==0
        plot(cg, -1, 'xk');
    end
    
    % - how mnay indep clusters
    % n = 0;
    
    % % add each SU
    % n = n + sum(labels==2);
    % % add merged MU
    % n = n + any(labels==1);
    
    % list_n_independent_clusters(end+1) = n;
    
    
    %     if ismember(1, labels) & ismember(2, labels)
    %         results(end)+1 = 2;
    %     elseif ismember(1, labels) & ismember(2, labels)
    list_labels{end+1} = labels;
end


if ~isempty(SAVEDIR_FINAL)
    fname = [SAVEDIR_FINAL '/final_clust_labels.png'];
    saveas(gcf, fname);
end

% PRINT SUMMARIES

disp('-------------');
tmp = [];
xs = {};

n = sum(cellfun(@(x)(ismember(1, x) & sum(ismember(x, 2))>1), list_labels));
disp(['N chans with MU + >1 SU: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = 'MU + >1 SU';

n = sum(cellfun(@(x)(ismember(1, x) & sum(ismember(x, 2))==1), list_labels));
disp(['N chans with MU + 1 SU: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = 'MU + 1 SU';

n = sum(cellfun(@(x)(~ismember(1, x) & sum(ismember(x, 2))>1), list_labels));
disp(['N chans with >1 SU: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = '>1 SU';

n = sum(cellfun(@(x)(~ismember(1, x) & sum(ismember(x, 2))==1), list_labels));
disp(['N chans with 1 SU: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = '1 SU';

n = sum(cellfun(@(x)(ismember(1, x) & sum(ismember(x, 2))==0), list_labels));
disp(['N chans with MU: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = 'MU';

n = sum(cellfun(@(x)(ismember(1, x) | sum(ismember(x, 2))>0), list_labels));
disp(['N chans with anything: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = 'anything';

n = sum(cellfun(@(x)(~ismember(1, x) & sum(ismember(x, 2))==0), list_labels));
disp(['N chans with nothing: ' num2str(n)]);
tmp(end+1) = n;
xs{end+1} = 'nothing';

inds_su = strcmp({DATSTRUCT.label_final}, 'su');
inds_mu = strcmp({DATSTRUCT.label_final}, 'mua');
inds_noise = strcmp({DATSTRUCT.label_final}, 'noise');

disp(['This many su: ' num2str(sum(inds_su))]);
tmp(end+1) = sum(inds_su);
xs{end+1} = 'N clust (SU)';

disp(['This many mu: ' num2str(sum(inds_mu))]);
tmp(end+1) = sum(inds_mu);
xs{end+1} = 'N clust (MU)';

disp(['This many noise: ' num2str(sum(inds_noise))]);
tmp(end+1) = sum(inds_noise);
xs{end+1} = 'N clust (noise)';

% disp(['This many independent clusters: ' num2str(sum(list_n_independent_clusters))]);

FigH = figure('visible', VISIBLE); hold on;
plot(1:length(tmp), tmp, '-ok');
for i=1:length(tmp)
    text(i, tmp(i), num2str(tmp(i)), 'color', 'r', 'fontsize', 12);
end
xticks(1:length(tmp));
xticklabels(xs);
ylabel('n chans with this');


if ~isempty(SAVEDIR_FINAL)
    fname = [SAVEDIR_FINAL '/final_num_clusts.png'];
    saveas(gcf, fname);
end
