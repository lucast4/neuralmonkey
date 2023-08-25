function datstruct_this = datstruct_merge_inner(datstruct)
    % merge them. They must be same chan.
    % PARAMS
    % - datstruct, array of structs.
    % RETURNS:
    % - datstruct_this, single struct.
    datstruct_this = struct;

    times_all =[];
    waveforms_all =[];
    amps_wf_all = [];
    for i=1:length(datstruct)
        times_all = [times_all; datstruct(i).times_sec_all];
        waveforms_all = [waveforms_all; datstruct(i).waveforms];
        amps_wf_all = [amps_wf_all; datstruct(i).amps_wf];
    end
    times_all = sort(times_all);

    datstruct_this.times_sec_all = times_all;
    datstruct_this.waveforms = waveforms_all;
    datstruct_this.amps_wf = amps_wf_all;

    % values that are the same
    for i=1:length(datstruct)
        for j=i+1:length(datstruct)
            assert(datstruct(i).batch == datstruct(j).batch);
            assert(datstruct(i).RSn == datstruct(j).RSn);
            assert(datstruct(i).chan == datstruct(j).chan);
            assert(strcmp(datstruct(i).label_final, datstruct(j).label_final));
            assert(datstruct(i).label_final_int == datstruct(j).label_final_int);
            assert(datstruct(i).chan_global == datstruct(j).chan_global);
        end
    end
    datstruct_this.batch = datstruct(1).batch;
    datstruct_this.RSn = datstruct(1).RSn;
    datstruct_this.chan = datstruct(1).chan;
    datstruct_this.label_final = datstruct(1).label_final;
    datstruct_this.label_final_int = datstruct(1).label_final_int;
    datstruct_this.chan_global = datstruct(1).chan_global;

    % values diff
    list_clust = [];
    list_index = [];
    for i=1:length(datstruct)
        list_clust(end+1) = datstruct(i).clust;
        list_index(end+1) = datstruct(i).index;
    end
    datstruct_this.clust_before_merge = list_clust;
    datstruct_this.index_before_merge = list_index;

    % Values ignored
    %         datstruct(1).clust_group_id;
    %         datstruct(1).clust_group_name;
    %         datstruct(1).GOOD;
    datstruct_this.clust = nan;
    datstruct_this.clust_group_id = nan;
    datstruct_this.clust_group_name = nan;
    datstruct_this.GOOD = 1;
    datstruct_this.times_sec_all_BEFORE_REMOVE_DOUBLE = nan;
    datstruct_this.index = nan;
end