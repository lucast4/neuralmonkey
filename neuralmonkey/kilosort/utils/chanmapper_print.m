LIST_BATCH = 1:4
nbatches = length(LIST_BATCH);
nchans_per_batch = 256/nbatches;
ct = 1;
chan_mapper = nan(3, nbatches, nchans_per_batch); % (rs, batch, chan_within_batch) --> chan_global.
for rs=2:3
    for batch=LIST_BATCH
        disp([rs, batch]);
        chans_withinbatch = 1:nchans_per_batch;
        chans_global = ct:ct+nchans_per_batch-1;
        ct=ct+nchans_per_batch;
        %         disp(chansthis)
        
        chan_mapper(rs, batch, chans_withinbatch) = chans_global;
        
        % print
        for i=1:length(chans_withinbatch)
            chan = chans_withinbatch(i);
            chang = chans_global(i);
            
%             disp(['ch ' num2str(chang) ' = rs ' num2str(rs) '-' num2str([batch chan])]);
%             chan_phy = chan-1; % 0-indexed
%             disp(['ch ' num2str(chang) ' = rs ' num2str(rs) ' ba ' num2str(batch) ' ch ' num2str(chan) ' ch_phy ' num2str(chan-1)]);
            disp(['ch ' num2str(chang) ' = r' num2str(rs) ' b' num2str(batch) ' chphy' num2str(chan-1) ' ch' num2str(chan)]);
        end
    end
end
