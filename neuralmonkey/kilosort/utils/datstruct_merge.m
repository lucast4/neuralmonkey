function DATSTRUCT_FINAL = datstruct_merge(DATSTRUCT, SAVEDIR_FINAL, ...
    LIST_MERGE_SU, indpeak, npre, npost, THRESH_SU_SNR, ...
    THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
    THRESH_ARTIFACT_ISI, MIN_SNR)
%% MERGE MU and split SU.
% DOes reclassify of SU after merging (either mu or su

if ~exist('LIST_MERGE_SU', 'var'); LIST_MERGE_SU = []; end
if ~exist('SAVEDIR_FINAL', 'var'); SAVEDIR_FINAL = []; end

list_chans_global = 1:512;
su_pairs_should_be_merged = {};

DATSTRUCT_FINAL = [];
for cg = list_chans_global
    
    %%%%%%%%%%%% Get all SU
    inds = [DATSTRUCT.chan_global]==cg & [DATSTRUCT.label_final_int]==2;
    datstruct = DATSTRUCT(inds);
    
    %     labels = [DATSTRUCT(inds).label_final_int];
    %     disp([cg labels]);
    
    %%% Sanity checks
    % 1) If multiple SUs, check their xcorr
    if false
        for i=1:length(datstruct)
            for j=i+1:length(datstruct)
                st1 = datstruct(i).times_sec_all;
                st2 = datstruct(j).times_sec_all;
                
                dt = 1/1000;
                [K, Qi, Q00, Q01, rir] = ccg(st1, st2, 500, dt);
                Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
                R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
                
                if Q<0.25 && R<0.05
                    % then this looks like same cluster...
                    % see l124 in splitAllClusters.m
                    su_pairs_should_be_merged{end+1} = [datstruct(i).index, datstruct(j).index];
                end
            end
        end
    end
    
    for i=1:length(datstruct)
        datstruct_this = datstruct(i);
        datstruct_this = rmfield(datstruct_this, 'sharpiness');
        datstruct_this = rmfield(datstruct_this, 'Q');
        datstruct_this = rmfield(datstruct_this, 'isi_violation_pct');
        datstruct_this = rmfield(datstruct_this, 'snr_final');
        datstruct_this = rmfield(datstruct_this, 'isbimod');
        datstruct_this.clust_before_merge = nan;
        datstruct_this.index_before_merge = nan;
        DATSTRUCT_FINAL = [DATSTRUCT_FINAL, datstruct_this];
    end
    
    %%%%%%%%%%%% Get all MU
    inds = [DATSTRUCT.chan_global]==cg & [DATSTRUCT.label_final_int]==1;
    datstruct = DATSTRUCT(inds);
    
    if length(datstruct)==0
        % do nothing
        %     elseif length(datstruct)==1
        %         DATSTRUCT_FINAL = [DATSTRUCT_FINAL, rmfield(datstruct, 'wf')];
    else
        % merge them
        datstruct_this = datstruct_merge_inner(datstruct);
        
        % Keep
        DATSTRUCT_FINAL = [DATSTRUCT_FINAL, datstruct_this];
    end
end

%% Merge SU if this is desired
if ~isempty(LIST_MERGE_SU)
    % 1) Create the merge.
    list_ds_append = [];
    for i=1:length(LIST_MERGE_SU)
        disp('++++++++++++++++++++++++++++++++++++++++++++++');
        %     ind1 = LIST_MERGE_SU{i}(1);
        %     ind2 = LIST_MERGE_SU{i}(2);
        inds_merge = LIST_MERGE_SU{i};
        
        % sanity check
        assert(length(unique(inds_merge)) == length(inds_merge));
        for j=1:length(inds_merge)
            assert(strcmp(DATSTRUCT(inds_merge(j)).label_final, 'su'));
        end

        disp('merging these..')
        disp(inds_merge);
        
        
        % do merge
        ds = datstruct_merge_inner(DATSTRUCT(inds_merge));
        
        % check if the merged is still su
        [ds, did_remove] = datstruct_remove_double_counted(ds, indpeak, npre, npost, false);
        
        if did_remove
            disp(['Merged cluster removed double spikes']);
        end
        DOPLOT = false;
        ds = datstruct_compute_metrics(ds, DOPLOT, ...
            indpeak, npre, npost);
        
        % remove label
        ds.label_final = [];
        ds.label_final_int = [];
        
        ds = datstruct_classify(ds, THRESH_SU_SNR, ...
            THRESH_SU_ISI, THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, ...
            THRESH_ARTIFACT_ISI, MIN_SNR);
        
        disp(ds);
        disp(['Merged cluster is classified as: ' ds.label_final]);
        
        % Plot combined waveforms
        n = size(ds.waveforms, 1);
        jump = round(n/100);
        wf = ds.waveforms(1:jump:n, :);
        figure; hold on;
        title(num2str(inds_merge));
        plot(wf');
        
        % Remove metrics, so can append
        ds = rmfield(ds, 'sharpiness');
        ds = rmfield(ds, 'Q');
        ds = rmfield(ds, 'isi_violation_pct');
        ds = rmfield(ds, 'snr_final');
        ds = rmfield(ds, 'isbimod');
        
        % Save it
        list_ds_append = [list_ds_append ds];
    end
    
    % 2) Remove the individual SUs
    list_inds_remove_from_DATSTRUCT_FINAL = [];
    for i=1:length(LIST_MERGE_SU)
        inds_merge = LIST_MERGE_SU{i};
        inds_remove = find(ismember([DATSTRUCT_FINAL.index], LIST_MERGE_SU{i}));
        assert(length(inds_remove)==length(LIST_MERGE_SU{i}))
        list_inds_remove_from_DATSTRUCT_FINAL = [list_inds_remove_from_DATSTRUCT_FINAL, ...
            inds_remove];
    end
    
    % 3) First remove, then append
    disp('Rwemovingn these inds from DATSTRUCT_FINAL:')
    disp(list_inds_remove_from_DATSTRUCT_FINAL);
    DATSTRUCT_FINAL(inds_remove) = [];
    
    disp('Appending this many ds to DATSTRUCT_FINAL:')
    disp(length(list_ds_append));
    DATSTRUCT_FINAL = [DATSTRUCT_FINAL, list_ds_append];
end

disp('This many indep clusters:');
disp(length(DATSTRUCT_FINAL));


end