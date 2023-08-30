function DATSTRUCT = datstruct_compute_metrics(DATSTRUCT, DOPLOT, ...
    indpeak, npre, npost, savedir)

% ================ CHANGE THIS
if ~exist('DOPLOT', 'var'); DOPLOT = false; end

% =========== OTHER STUFF.
if DOPLOT
    % close all;
    savedir = [savedir '/clust_chan_mappings_figures/raw_and_snr'];
    mkdir(savedir);
end
% indpeak = -gwfparams.wfWin(1);

% Collect for plotting distributions
list_sharpiness = [];
list_refract = [];
list_isi = [];
list_snr = [];
list_clust_group_name = {};
list_GOOD = [];
for i=1:length(DATSTRUCT)
    disp(i)
    waveforms = DATSTRUCT(i).waveforms;
    rs = DATSTRUCT(i).RSn;
    batch = DATSTRUCT(i).batch;
    clust = DATSTRUCT(i).clust;
    chan = DATSTRUCT(i).chan;
    GOOD = DATSTRUCT(i).GOOD;
    %     clust_group_name = DATSTRUCT(i).clust_group_name;
    
    % sharpiness
    sharpiness = sharpiness_compute(waveforms, false);
    
    st = DATSTRUCT(i).times_sec_all;
    [Q, R, isi_violation_pct] = refractoriness_compute(st);

    % 1) remove outliers [SKIP]
    % 5) align (use xcorr?) [check 2-2-182]
    [snr_final, snr_new, peak_to_trough, waveforms_running_std, snr_old, ...
        peak_to_trough_old, waveforms_running_std_old, waveforms_aligned] = ...
        snr_compute_wrapper(waveforms, indpeak, npre, npost);
    
    % Is this combo of pos and negative? if so then compute snr separately
    % then average
    [isbimod, waveforms_pos, waveforms_neg] = is_bimodal_waveform(waveforms, indpeak);
    if isbimod
        % Try splitting and computing snr
        [snr_pos] = snr_compute_wrapper(waveforms_pos, indpeak, npre, npost);
        [snr_neg] = snr_compute_wrapper(waveforms_neg, indpeak, npre, npost);
        
        disp(['After splitting (old, pos, neg): ' num2str(snr_final) ', ' num2str(snr_pos) ', ' num2str(snr_neg)]);
        % Take the split snrs if this improves the snr.
        snr_final = max(mean([snr_pos, snr_neg]), snr_final);
    end
    
    %%% COLLECT
    list_sharpiness(end+1) = sharpiness;
    list_refract(end+1) = Q;
    list_isi(end+1) = isi_violation_pct;
    list_snr(end+1) = snr_final;
    list_GOOD(end+1) = GOOD;
    
    DATSTRUCT(i).sharpiness = sharpiness;
    DATSTRUCT(i).Q = Q;
    DATSTRUCT(i).isi_violation_pct = isi_violation_pct;
    DATSTRUCT(i).snr_final = snr_final;
    DATSTRUCT(i).isbimod = isbimod;
    
    if DOPLOT
        % 6) split pos and neg going before doing this.
        FigH = figure('Position', get(0, 'Screensize'), 'visible', 'off'); hold on;
        n = min([size(waveforms,1), 100]);
        
        subplot(2,2,1); hold on;
        title(['orig. not shifted; refract: ' num2str(Q)]);
        plot_waveforms_singleplot(waveforms, n);
        line([indpeak-npre indpeak-npre], ylim)
        line([indpeak+npost indpeak+npost], ylim)
        
        subplot(2,2,2); hold on;
        nthis = size(waveforms_aligned,1);
        plot(waveforms_aligned(1:nthis, :)');
        title(['shifted. this used for computing snr. snr_final ' num2str(snr_final)]);
        
        subplot(2,2,3); hold on;
        title(['orig: std (blue) and peak-to-trough (red): ' num2str(snr_old)]);
        plot(1:length(waveforms_running_std_old), waveforms_running_std_old);
        plot([0, length(waveforms_running_std_old)], [peak_to_trough_old, peak_to_trough_old]);
        line([indpeak-npre indpeak-npre], ylim)
        line([indpeak+npost indpeak+npost], ylim)
        line(xlim, [0 0]);
        
        subplot(2,2,4); hold on;
        title(['shifted: std (blue) and peak-to-trough (red) ' num2str(snr_new)]);
        plot(1:length(waveforms_running_std), waveforms_running_std);
        plot([0, length(waveforms_running_std)], [peak_to_trough, peak_to_trough]);
        line(xlim, [0 0]);
        
        fname = [savedir '/' 'rs_' num2str(rs) '-ba_' num2str(batch) '-ch_' num2str(chan) '-cl_' num2str(clust ) '-idx_' num2str(i) '-snr_' num2str(snr_final) '-Q_' num2str(Q) '-shrp_' num2str(sharpiness) '.png'];
        disp(fname)
        saveas(gcf, fname);
        
        % Also save with snr at front, for easy sorting.
        fname = [savedir '/' 'snr_' num2str(snr_final) '-rs_' num2str(rs) '-ba_' num2str(batch) '-ch_' num2str(chan) '-cl_' num2str(clust ) '-idx_' num2str(i) '-Q_' num2str(Q) '-shrp_' num2str(sharpiness) '.png'];
        disp(fname)
        saveas(gcf, fname);
        
        if false
            % Also save with Q at front, for easy sorting.
            fname = [savedir '/' 'Q_' num2str(Q) '-rs_' num2str(rs) '-ba_' num2str(batch) '-ch_' num2str(chan) '-cl_' num2str(clust ) '-idx_' num2str(i) '-snr_' num2str(snr_final) '-shrp_' num2str(sharpiness) '.png'];
            disp(fname)
            saveas(gcf, fname);

            % Also save with sharpiness at front, for easy sorting.
            fname = [savedir '/' 'shrp_' num2str(sharpiness) '-rs_' num2str(rs) '-ba_' num2str(batch) '-ch_' num2str(chan) '-cl_' num2str(clust ) '-idx_' num2str(i) '-snr_' num2str(snr_final) '-Q_' num2str(Q) '.png'];
            disp(fname)
            saveas(gcf, fname);
        end
        close all
    end
end
