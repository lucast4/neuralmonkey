function st_new = spiketimes_remove_double_counted(st, THRESH_CLOSE)
	% - THRESH_CLOSE, in sec, if two spikes are this close, then keeps the second one.
	% THRESH_CLOSE = 0.00025;

	if ~exist('THRESH_CLOSE', 'var'); THRESH_CLOSE = 0.00025; end
	st_new = [];
	for i=1:length(st)-1
	    t1 = st(i);
	    t2 = st(i+1);
	    
	    if t2-t1>THRESH_CLOSE
	    	% Keep, otherwise throw out the first 
	        st_new(end+1) = t1;
	    end
	end
	% add the last, since it is not checked above.
	st_new(end+1) = st(end);

	if false
		disp(['Frac spikes kept: ' num2str(length(st_new)/length(st))]);
	end

	st_new = st_new';
end