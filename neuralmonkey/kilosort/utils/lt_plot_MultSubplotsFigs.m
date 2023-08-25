%% LT 7/20/15 - gives you correct figure and subplot number, so you can put multiple subplots over multiple figs

function [fignums_alreadyused, hfigs, count, hsplot, fignum, ...
	subplot_num]=lt_plot_MultSubplotsFigs(SubplotsPerFig, subplotrows, ...
		subplotcols, fignums_alreadyused, hfigs, count, INVISIBLE);


% To run:

% Params:
% figcount = 1 % Put this OUTSIDE the for loop
% SubplotsPerFig - how many subplots per fig?
% subplotrows and subplotcols should match SubplotsPerFig
% fignums_alreadyused=[];
% hfigs=[];

% EXAMPLE:
% figcount=1;
% subplotrows=2;
% subplotcols=3;
% fignums_alreadyused=[];
% hfigs=[];
% hsplots = [];

% [fignums_alreadyused, hfigs, figcount, hsplot]=lt_plot_MultSubplotsFigs('', subplotrows, subplotcols, fignums_alreadyused, hfigs, figcount);


% put this function where you would normally put subplot
% put the params above before the entire for loop.

% This code automatically pulls up the correct figure and subplot.  (i.e.
% no need to call figure(fignum) and subplot(subplotnum);

%% PARAMS

if ~exist('INVISIBLE', 'var'); INVISIBLE = false; end

% fignums_alreadyused=[];
% hfigs=[];
SubplotsPerFig=subplotrows*subplotcols;

%% RUN

fignum=ceil(count/SubplotsPerFig);

% Keep track of all fignums alraedy used
fignums_alreadyused=[fignums_alreadyused fignum];

% if this is new figure, get a new handle
if length(fignums_alreadyused)==1 || fignums_alreadyused(end)~=fignums_alreadyused(end-1);
    % then is new figure
    if INVISIBLE
    	hfigs(fignum) = figure('Position', get(0, 'Screensize'), 'visible','off'); hold on; % full screen
	else
    	hfigs(fignum) = figure('Position', get(0, 'Screensize')); hold on; % full screen
    end
    % hfigs(fignum)=figure; hold on;
    lt_plot_format;
else
    % is old figure, just reopen it
    if INVISIBLE
    	set(0,'CurrentFigure',hfigs(fignum))
    	% figure(hfigs(fignum), 'visible','off');
	else
    	figure(hfigs(fignum));
	end
end

% Which subplot is this?
subplot_num=count-(fignum-1)*SubplotsPerFig;
hsplot=lt_subplot(subplotrows, subplotcols, subplot_num); hold on;
set(gca, 'YColor', [0.5 0.5 0.5])
set(gca, 'XColor', [0.5 0.5 0.5])
try
    subplotsqueeze(gca, 1.07);
catch err
end
% Update count
count=count+1;




