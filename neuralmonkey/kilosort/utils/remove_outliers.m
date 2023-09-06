function [vals_new, inds_remove] = remove_outliers(vals, n_iqr, PLOT)

% Detect and remove outliers in the arrya of sclaars.
% Default is to use detection that maximum throws out 2.5% of data.
% Is like tukey, but uses wrider range for iqr to allow for skewed data.
% Is conservative.
% PARAMS
% - n_iqr, multiple of iqr to add to low and high, to make bounds.
% RETURNS:
% - vals_new, reduced array, after removign outlers
% - inds_remove, logical array.

if ~exist('PLOT', 'var'); PLOT = false; end
if ~exist('n_iqr', 'var'); n_iqr = 3; end

if false
    % Traditional tukey
    bounds_25_75 = prctile(vals, [25 75]);
    iqr = (bounds_25_75(2) - bounds_25_75(1));
    
    n_iqr = 3;
    low = bounds_25_75(1)-n_iqr*iqr;
    high = bounds_25_75(2)+n_iqr*iqr;
else
    % Better for skewed dta
    bounds_25_75 = prctile(vals, [2.5 97.5]);
    iqr = (bounds_25_75(2) - bounds_25_75(1));
    
    n_iqr = 2;
    low = bounds_25_75(1)-n_iqr*iqr;
    high = bounds_25_75(2)+n_iqr*iqr;
end

inds_remove = (vals<low) | (vals>high);

if any(inds_remove) & PLOT
    disp(['REMOVING THESE INDS: ']);
    disp(find(inds_remove));
    % disp(['IQR bounds: ' num2str(bounds_25_75)]);
    % disp([low high]);
    figure; hold on;
    subplot(1,2,1); hold on;
    title('outliers(r)')
    plot(vals, 1, 'xk');
    plot(bounds_25_75, 1, 'db');
    line([low low], ylim);
    line([high high], ylim);
    plot(vals(inds_remove), 1, 'or')
end

vals_new = vals;
vals_new(inds_remove) = [];

end
