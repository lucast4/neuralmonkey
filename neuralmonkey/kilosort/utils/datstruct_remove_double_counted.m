function [DATSTRUCT, DID_REMOVE] = datstruct_remove_double_counted(DATSTRUCT, indpeak, ...
    npre, npost, DRYRUN)
	%% Find double-counted waveforms
	% if refrac violations are really high within short time...
    if ~exist('DRYRUN', 'var'); DRYRUN = false; end
    
	disp('----');
	PRINT = false;
	DID_REMOVE = false;
	for i=1:length(DATSTRUCT)
	    st = DATSTRUCT(i).times_sec_all;
	    waveforms =DATSTRUCT(i).waveforms;

	    % threshold = 0.0005; % second, to call it a violatioin.

	    % st = sort(st);
	    % isi = diff(st);

	    % THRESH_CLOSE = 0.00025;
	    % THRESH_FAR = 0.002;
	    % THRESH_BASE = 0.02;
	    % violations_close = isi<THRESH_CLOSE;
	    % violations_far = (isi<THRESH_FAR) & (isi>THRESH_CLOSE);
	    % violations_base = (isi<THRESH_BASE) & (isi>THRESH_FAR);

	    % rate_close = sum(violations_close)/(THRESH_CLOSE - 0);
	    % rate_far = sum(violations_far)/(THRESH_FAR - THRESH_CLOSE);
	    % rate_base = sum(violations_base)/(THRESH_BASE - THRESH_FAR);
	    
	    % if rate_far==0
	    %     rate_far = rate_base;
	    % end
	    
	    % double_count_rate = rate_close/rate_far;
	    % double_count_rate_relbase = rate_close/rate_base;
	    % rate_far_rel_base = rate_far/rate_base;
	    
	    % % estimate frac of total spiies that are double counted
	    % n_double = sum(violations_close);
	    % n_total = length(st);
	    % frac_spikes_double_counted = n_double/n_total;

		[double_count_rate, double_count_rate_relbase, rate_far_rel_base, ...
			frac_spikes_double_counted] = datstruct_remove_double_counted_inner(st);

	    % snr
	    [snr_this, snr_new, ~, ~, snr_old]= snr_compute_wrapper(waveforms, indpeak, npre, npost);
	    % how much is snr improved by shifting
	    snr_improvement = snr_new/snr_old;
	    
	    if PRINT
            disp([i, double_count_rate, double_count_rate_relbase, rate_far_rel_base, frac_spikes_double_counted, snr_this, snr_improvement]);
	    end
	    
	    % criteria to call remove the close spikes
	    crit1 = double_count_rate>5 & rate_far_rel_base<0.25 & snr_this>4 & snr_improvement>2 & double_count_rate_relbase/rate_far_rel_base>5;
	    crit2 = double_count_rate>10 & rate_far_rel_base<0.75 & snr_this>4 & snr_improvement>1 & double_count_rate_relbase/rate_far_rel_base>5;
	    crit3 = double_count_rate>30 & rate_far_rel_base<3 & snr_this>4 & snr_improvement>1 & double_count_rate_relbase/rate_far_rel_base>5;
	    crit4 = double_count_rate>50 & rate_far_rel_base<5 & snr_this>4 & snr_improvement>1 & double_count_rate_relbase/rate_far_rel_base>5;

	    if crit1 | crit2 | crit3 | crit4
	        % Keep the first spike for any that are double counted
	        
	        disp(['Removing double spikes for ' num2str(i)]);
	        
	        st_new = spiketimes_remove_double_counted(st);
	        
            if DRYRUN
                disp('DRY RUN, skipping...');
            else
                DATSTRUCT(i).times_sec_all_BEFORE_REMOVE_DOUBLE = DATSTRUCT(i).times_sec_all;
                DATSTRUCT(i).times_sec_all = st_new;

                DID_REMOVE = true;
            end
	        disp(['... frac spikes kept: ' num2str(length(st_new)/length(st))]);
	    end            
	end
end