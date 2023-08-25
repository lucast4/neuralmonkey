function [inds_out, inds_unique, X_cell] = lt_tools_grp2idx(X)
%%
% X is cell array (1xn, where n is dim). each entyr should be column opf
% str or numbers
% % e.g. X =
% 
%   1×2 cell array
% 
%     [47×1 char]    [47×1 char]
% note, columns can be strings or num arrays.

% OUT:
% grp indices
%%
% === first convert all things to strings
for j=1:length(X)
    if isnumeric(X{j})
        X{j} = num2str(X{j});
    end
end

X_str = [];
for j=1:length(X)
    X_str = strcat(X_str, '@', X{j}); % put at sign so that dont get fluke similarityes (e.g. [11 0] woudl be asme as [1 10])
end

inds_out = grp2idx(X_str);
inds_unique = unique(inds_out);

X_cell = mat2cell(X_str, ones(size(X_str,1),1), size(X_str,2)); % for outpe, Nx1 cell.
