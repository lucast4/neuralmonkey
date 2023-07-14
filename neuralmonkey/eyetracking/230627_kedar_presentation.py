# move everything 
# then change so it's independent of session, uses XY data

from neuralmonkey.classes.session import load_mult_session_helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import math

################################################
### Extract a specific session ###
################################################
def __main__():
	# preprocessed datasets: Diego-230603 (ok), Pancho-221020 (ok), Diego-230626 (primsingrid), Diego-230616 (26 singleprims)
	date = 230616
	animal = "Diego"
	session = 0

	MS = load_mult_session_helper(date, animal)
	sn = MS.SessionsList[session]

################################################
### Plot a single trial's stimulus/behavior ###
################################################

# plot the task image for a trial
def plotTrialTaskImage(sn, trial, ax=None):
    if ax==None:
        fig, ax = plt.subplots()
    sn.plot_taskimage(ax, trial)
    return ax

# plot the (x,y) data of all drawn strokes
def plotTrialDrawnStrokes(sn, trial, ax=None):
    # get list of strokes and times
    strk = sn.strokes_extract(trial, peanuts_only=True) # NOTE: peanuts_only=True pulls out only drawing strokes (e.g. not fixation etc)
    
    # if no specified ax, then default to overlaying onto task image
    if ax==None:
        ax = plotTrialTaskImage(sn, trial)
        
    # now overlay each stroke
    for s in strk:
        ax.scatter(s[:,0], s[:,1], c='cyan', marker=',', alpha=0.15)
        
    return ax

################################################
### Extracting a Session's successful Trialcodes/Trialnums ###
################################################

# get a list of successful trialcodes
def getSuccessfulTrialCodes(sn):
    D = sn.Datasetbeh
    Dcopy = D.copy()
    Dcopy.preprocessGood(params=["one_to_one_beh_task_strokes"]) # prunes Dcopy to keep only successful trials
    
    return Dcopy.Dat['trialcode'].tolist()

# from the list of trialcodes, get matching list of NEURAL trials
def getTrialNumsFromTrialCodes(sn, trialcode_list):
    D = sn.Datasetbeh
    trials = []
    for tc in trialcode_list:
        # make sure there is exactly one matching trialcode
        #index_dataset = D.Dat[(D.Dat["trialcode"]==tc)].index
        #assert len(index_dataset)==1

        # pull out these datapoints from trial-level dataset
        t = sn.datasetbeh_trialcode_to_trial(tc)
        trials.append(t)
        #ons, offs = sn.strokes_extract_ons_offs(trialtdt)
    return trials

# get a list of successful trialnums
def getSuccessfulTrialNums(sn):
    trialcodes = getSuccessfulTrialCodes(sn)
    return getTrialNumsFromTrialCodes(sn, trialcodes)

################################################
### Extracting TaskStroke Tokens/Coordinates From [Session, Trial]
################################################

# define TaskStroke object which represents a TaskStroke and stores token/coordinates
class TaskStroke:
    def __init__(self, token, coords):
        self.token = token
        self.coords = coords
        self.x_coords = self.coords[:, 0]
        self.y_coords = self.coords[:, 1]

# array of tokens, each one is a task stroke with info such as shapename etc.
def _getAllTaskStrokeTokens(sn, trial):
    dataset_index_from_neural = sn.datasetbeh_trial_to_datidx(trial)
    return sn.Datasetbeh.taskclass_tokens_extract_wrapper(dataset_index_from_neural, "task", plot=False)

# array of arrays of XY data, each one is the many points making up a single task stroke
def _getAllTaskStrokeCoordinates(sn, trial):
    return sn.strokes_task_extract(trial)

# get tokens, coordinates for all task strokes
def getAllTaskStrokes(sn, trial):
    tokens = _getAllTaskStrokeTokens(sn, trial)
    coords = _getAllTaskStrokeCoordinates(sn, trial)
    
    if len(tokens) != len(coords):
        print("t", len(tokens))
        print("c", len(coords))
        assert False
    
    task_strokes = []
    for i in range(len(tokens)):
        t = TaskStroke(tokens[i], coords[i])
        task_strokes.append(t)
    
    if len(task_strokes)==0:
        assert False
    return task_strokes

# gets the token, coords for a single task stroke (specified by index)
def getSingleTaskStroke(sn, trial, task_stroke_index):
    task_strokes = getAllTaskStrokes(sn, trial)
    return task_strokes[task_stroke_index]

################################################
### Extracting BehStroke Tokens/Coordinates from [Session, Trial] ###
################################################

# define BehStroke object which represents a BehStroke and stores token/coordinates
class BehStroke:
    def __init__(self, token, coords):
        self.token = token
        self.coords = coords
        self.x_coords = self.coords[:, 0]
        self.y_coords = self.coords[:, 1]

# array of tokens, each one is a beh stroke with info such as shapename etc.
def _getAllBehStrokeTokens(sn, trial):
    dataset_index_from_neural = sn.datasetbeh_trial_to_datidx(trial)
    return sn.Datasetbeh.taskclass_tokens_extract_wrapper(dataset_index_from_neural, "beh_firsttouch", plot=False)

# array of arrays of XY data, each one is the many points making up a single beh stroke
def _getAllBehStrokeCoordinates(sn, trial):
    return sn.strokes_extract(trial)

# get tokens, coordinates for all task strokes
def getAllBehStrokes(sn, trial):
    tokens = _getAllBehStrokeTokens(sn, trial)
    coords = _getAllBehStrokeCoordinates(sn, trial)
    
    beh_strokes = []
    for i in range(len(tokens)):
        t = BehStroke(tokens[i], coords[i])
        beh_strokes.append(t)
    
    if len(beh_strokes)==0:
        assert False
    return beh_strokes

################################################
### Extracting/Plotting Eye-Tracking Data ###
################################################

# plot raw eye-tracking data (not calibrated, so may be much smaller than task image)
def plotRawEyeXY(sn, trial):
    # get RAW x-coordinates (not transformed by calibration matrix)
    times_x, vals_x, fs_x = sn.extract_data_tank_streams("eyex", trial, ploton=False)

    # get RAW y-coordinates (not transformed by calibration matrix)
    times_y, vals_y, fs_y = sn.extract_data_tank_streams("eyey", trial, ploton=False)

    # plot RAW xy-coordinates
    fig, ax = plt.subplots()
    ax.scatter(vals_x, vals_y)

# plot calibrated/smoothed eye-tracking data over task image
def plotSmCalibEyeXYOverTaskImage(sn, trial):
    x, y, times = getEyeXYSmoothedAndTransformed(sn, trial, PLOT=False)

    # plot RAW xy-coordinates
    fig, ax = plt.subplots()
    plotTrialTaskImage(sn, trial, ax)
    ax.scatter(x, y)

# returns smoothed and transformed x,y data for a session/trialnum
def getEyeXYSmoothedAndTransformed(sn, trialnum, PLOT=True):
    # get TRANSFORMED xy-coordinates (used calibration matrix to map to screen)
    times = sn.beh_extract_eye_good(trialnum)[0]
    x_aff = sn.beh_extract_eye_good(trialnum)[1][:,0]
    y_aff = sn.beh_extract_eye_good(trialnum)[1][:,1]

    # SMOOTH DATA
    from pythonlib.pythonlib.tools.timeseriestools import smoothDat
    x_aff_sm = smoothDat(x_aff, window_len=10)
    y_aff_sm = smoothDat(y_aff, window_len=10)
    
    if PLOT==True:
        fig, axes = plt.subplots(2,1, figsize=(10, 10))
        # plot smoothed x-data
        axes[0].plot(times,x_aff_sm)
        # plot smoothed y-data
        axes[1].plot(times,y_aff_sm)
        return x_aff_sm, y_aff_sm, times, axes
    else:
        return x_aff_sm, y_aff_sm, times

# plot the (x,y) data over time for eye tracking data
def plotEyeTrackingTrace(times, x_vals, y_vals, ax, color="b"):
    ax.plot(x_vals, y_vals, "-k", alpha=0.3)
    ax.scatter(x_vals, y_vals, c=times, alpha=0.5, marker="o")   

###################################################
### Extracting Fixations from Eye-Tracking Data ###
###################################################

# define Fixation object which represents an entire "fixation event"
class Fixation:
    def __init__(self, trialnum, x_vals, y_vals, times):
        assert len(x_vals) == len(y_vals)
        assert len(x_vals) == len(times)
        
        self.trialnum = trialnum
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.centroid = _getCentroidOfXY(self.x_vals, self.y_vals)
        self.times = times
        self.onset = times[0]
        self.offset = times[-1]
        self.task_stroke = None

# @params x_vals, y_vals: SMOOTHED XY-data
# @return array of len(x_vals)==len(y_vals)
def getAllFixationEventsFromEyeXY(trialnum, x_vals, y_vals, times, window=5, threshold=5, min_fixation_length=100):
    assert len(x_vals) == len(y_vals)
    assert len(x_vals) == len(times)
    
    # note: len(x_diff) == len(x_vals)-1, so will have to insert dummy value at index 0
    x_diff = np.diff(x_vals)
    y_diff = np.diff(y_vals)

    results = [None] * len(x_diff)
    
    # NOTE: could possibly improve by using instantaneous velocity
    for i in range(len(x_diff)-window):
        x_win = abs(np.sum(x_diff[i:i+window]))
        y_win = abs(np.sum(y_diff[i:i+window]))
        if x_win <= threshold and y_win <= threshold:
            for j in range(i, i+window):
                results[j] = 'fixation'
        else:
            for j in range(i, i+window):
                results[j] = 'saccade'
        
    # insert dummy value
    results.insert(0, results[0])
    
    # make Fixation objects by finding stretches of fixations that are above min_fixation_length
    fixations = []
    i = 0
    while i<len(results):
        if results[i] == 'fixation':
            j = i+1
            while j < len(results) and results[j]=='fixation':
                j=j+1
            if j-i >= min_fixation_length:
                # create Fixation object
                #print(trialnum, x_vals[i:j], y_vals[i:j], times[i:j])
                f = Fixation(trialnum, x_vals[i:j], y_vals[i:j], times[i:j])
                fixations.append(f)
            i = j
        else:
            i = i+1
        
    return np.array(fixations)

# get fixations within time window (used to find if fixation is between events)
# note: he might still be looking at something after event ends, what's more important
# is whether he starts looking at it within the time window
def getFixationsBetweenEvents(sn, trial, fixations, start_event, end_event):
    start_time, end_time = _getTimeWindowOfEvents(sn, trial, start_event, end_event)
    results = []
    for f in fixations:
        if f.onset >= start_time and f.onset <= end_time:
            results.append(f)
            
    return results

# get the start, end times for the window spanned by start_event, end_event
def _getTimeWindowOfEvents(sn, trial, start_event, end_event):
    # keep just times between [start_event, end_event]
    dict_event_times = sn.events_get_time_sorted(trial, list_events=(start_event, end_event))[0]
    start_time = dict_event_times[start_event]
    end_time = dict_event_times[end_event]
    
    return start_time, end_time

# Get the centroid of many (x,y) value pairs
def _getCentroidOfXY(x_vals, y_vals):
    return [np.mean(x_vals), np.mean(y_vals)]

# get the centroids of all task strokes (e.g. shapes, usually) in a session's trial
# NOTE: index of returned array will be in particular order, same as tokens
def getAllTaskStrokeCentroids(sn, trial):
    # get tokens, coordinates for all task strokes in this trial
    task_strokes = getAllTaskStrokes(sn, trial)
    
    # get centroid of each task stroke
    n_strks = len(task_strokes)
    centroids = [None] * n_strks
    
    for i in range(n_strks):
        # get task stroke coordinates
        t = task_strokes[i]
        # calculate task stroke centroid
        centroid = _getCentroidOfXY(t.x_coords, t.y_coords)
        centroids[i] = centroid
    
    return centroids

# get the token/coordinates of the task stroke whose centroid is closest to the given fixation
# NOTE: if failing often, lower maxdist (200 is hand-picked and somewhat arbitrary)
def getClosestTaskStrokeToFixation(sn, fixation, maxdist=200):
    # get trialnum and all task_strokes
    trial = fixation.trialnum
    task_strokes = getAllTaskStrokes(sn, trial)
    
    # get fixation, task_stroke centroids
    fixation_centroid = fixation.centroid
    task_stroke_centroids = getAllTaskStrokeCentroids(sn, trial)
    
    # calculate distance from fixation centroid to each task stroke centroid
    dists = [None] * len(task_strokes)
    
    for i in range(len(task_stroke_centroids)):
        task_stroke_centroid = task_stroke_centroids[i]
        dists[i] = math.dist(fixation_centroid, task_stroke_centroid) # cartesian distance
    
    # get index of closest task stroke
    closest_task_stroke_ind = np.argmin(dists)
    
    # if below dist threshold, store in f
    if dists[closest_task_stroke_ind] <= maxdist:
        return task_strokes[closest_task_stroke_ind]
    else:
        return None # meaning too far from any shape


# MAIN FUNCTION TO GET FIXATIONS
# returns array of Fixations which store a task_stroke that it corresponds to
def getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event, keep_fixations_off_shape=False, maxdist=150):
    vals_x, vals_y, times = getEyeXYSmoothedAndTransformed(sn, trial, PLOT=False)
    fixations = getAllFixationEventsFromEyeXY(trial, vals_x, vals_y, times)
    
    fx = []
    # find what shape he's fixating on, for each fixation
    for f in fixations:
        ts = getClosestTaskStrokeToFixation(sn, f)
        f.task_stroke = ts
        
        if ts==None:
            if keep_fixations_off_shape==True:
                fx.append(f)
        else:
            fx.append(f)
        
    # slice out only the fixations between two events
    fxs = getFixationsBetweenEvents(sn, trial, fx, start_event, end_event)
    return fxs

##########################
### Plotting Fixations ###
##########################

# make a plot of what shape he is fixating on, over the course of a trial
# NOTE: shapes are represented by their taskstroke index (0-indexed)
def plotShapeFixationOverTime(sn, trial, start_event, end_event):
    # get matched fixations
    fixations = getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event)
    # get the midpoint time for each fixation (so we can treat as single timepoint)
    fix_times = [np.mean(fixations[i].times) for i in range(len(fixations))]
    # get shape_inds, shape_names and store as dictionary
    fix_taskstroke_inds = [fixations[i].task_stroke.token['ind_taskstroke_orig'] for i in range(len(fixations))]
    fix_taskstroke_shapenames = [fixations[i].task_stroke.token['shape'] for i in range(len(fixations))]
    shape_dict = [(fix_taskstroke_inds[i], fix_taskstroke_shapenames[i]) for i in range(len(fix_taskstroke_inds))]

    # plot
    fig, ax = plt.subplots()
    ax.scatter(fix_times, fix_taskstroke_inds, alpha=0.5, marker="o")
    return ax, shape_dict
    
# plot fixation events over task image
def plotFixationsOverTaskImage(sn, trial, start_event, end_event, keep_fixations_off_shape=False):
    from pythonlib.tools.plottools import makeColors
    
    # get fixations
    fixations = getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event, keep_fixations_off_shape)
    # plot task image
    ax = plotTrialTaskImage(sn, trial)
    # get color gradient for timepoints
    pcols = makeColors(len(fixations))
    
    # plot each fixation
    for i in range(len(fixations)):
        f = fixations[i]
        col = pcols[i]
        ax.scatter(f.x_vals, f.y_vals, color=col)
    return ax
    
# plot heatmap of fixations for trial
def plotFixationHeatMapForTrial(sn, trial, start_event, end_event, keep_fixations_off_shape=False):
    # get fixations
    fixations = getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event, keep_fixations_off_shape)
    
    # make a 2x1 plot grid, first plot will be raw task image, second will be heatmap
    fig, axes = plt.subplots(1,2)
    plotTrialTaskImage(sn, trial, axes[0])
    
    # add each fixation to heatmap
    for i in range(len(fixations)):
        f = fixations[i]
        axes[1].hist2d(f.x_vals, f.y_vals,bins=40)
    
    return fig,axes

### Extracting all ShapeNames, Locs in Session ###

# get all shape names for a given session
def getAllShapeNamesForSession(sn, D, DS):
	#D = sn.Datasetbeh;
	#DS = DatStrokes(D) # DS holds all of a session's strokes
    # extract Dataset for this session
    from pythonlib.dataset.dataset_strokes import DatStrokes

    # pull out all shape names for session
    list_shapes = DS.Dat["shape"].unique()
    return list_shapes

# get all locs for a given session
def getAllLocsForSession(sn, D, DS):
	#D = sn.Datasetbeh;
	#DS = DatStrokes(D) # DS holds all of a session's strokes
    # extract Dataset for this session
    from pythonlib.dataset.dataset_strokes import DatStrokes

    # pull out all shape names for session
    list_locs = DS.Dat["gridloc"].unique()
    return list_locs

#########################################################################
### Extracting all Trialcodes, Trialnums for a given [shapename, loc] ###
#########################################################################

# get all trialcodes for a given shape
def getAllTrialCodesForShapeAndLocInSession(sn, D, DS, shapename, loc):
    s_all_trialcodes = DS.Dat[(DS.Dat["shape"]==shapename) & (DS.Dat["gridloc"]==loc)]["dataset_trialcode"].tolist()
    return s_all_trialcodes

# get all trialnums for a given shape
def getAllTrialNumsForShapeAndLocInSession(sn, D, DS, shapename, loc):
    s_all_trialcodes = getAllTrialCodesForShapeAndLocInSession(sn, D, DS, shapename, loc)
    return getTrialNumsFromTrialCodes(sn, s_all_trialcodes)

###################################################
### Plotting Skeleton of Looking Behavior for Trial ###
#######################################################

# plot skeleton for single trial (e.g. connect midpoints of each fixation), overlaid onto task image
def plotSkeletonForTrial(sn, trial, start_event, end_event, ax=None, keep_fixations_off_shape=False, task_img_overlay=True):
    from pythonlib.tools.plottools import makeColors
    
    # get fixations
    fixations = getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event, keep_fixations_off_shape)
    # get fixation centroids
    f_centroids = np.array([f.centroid for f in fixations])
    x = [f_centroids[i][0] for i in range(len(f_centroids))]
    y = [f_centroids[i][1] for i in range(len(f_centroids))]

    # get colors for indicating points order
    cols = makeColors(len(x), cmap='viridis')
    
    if ax==None:
        fig,ax = plt.subplots()
        
    # plot lines between points
    ax.plot(x, y, '-', alpha=0.5)
    # plot actual points
    ax.scatter(x, y, c=cols)
    # set title as trial num
    ax.set_title('trial: ' + str(trial))
    
    # plot task image
    if task_img_overlay:
        plotTrialTaskImage(sn, trial, ax)
    return ax

###################################################################
### Plotting Spaghetti Skeleton of Looking Behavior on TrialList ###
###################################################################

# plot all skeletons for a single shape, in individual subplots
def plotSkeletonsForTrialList(sn, trials, start_event, end_event, keep_fixations_off_shape=False, subplot=False):
    n_trials = len(trials)
    
    if n_trials==0:
        assert False

    if subplot==True:
        fig, axes = plt.subplots(n_trials, 1, figsize=(10, 3*n_trials))

        for i in range(n_trials):
            t = trials[i]
            plotSkeletonForTrial(sn, t, start_event, end_event, axes[i], keep_fixations_off_shape)

        return fig, axes
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        for i in range(n_trials):
            t = trials[i]
            plotSkeletonForTrial(sn, t, start_event, end_event, ax, keep_fixations_off_shape)
        
        return fig, ax

##########################################################################
### Saving .PNG figures of Skeletons for all shapes/locs/event_windows ###
##########################################################################

# Make 3-levels of plots
# - for each shape:
# -- for each location:
# --- for each of 2 event windows (before_drawing, during_drawing):
# ---- for each value of subplots (True, False)
def makeSkeletonPNGS(sn):
	from pythonlib.dataset.dataset_strokes import DatStrokes

	D = sn.Datasetbeh;
	DS = DatStrokes(D) # DS holds all of a session's strokes

	# get all shapes
	shape_list = getAllShapeNamesForSession(sn, D, DS)
	windows = [["stim_onset", "go"], ["go", "off_stroke_last"]]
	subplots = [True, False]

	for shape in shape_list:
	    # get all locs this shape appeared in
	    DS_shape = DS.Dat[DS.Dat['shape']==shape]
	    shape_locs = DS_shape['gridloc'].unique().tolist()
	        
	    for loc in shape_locs:
	        for window in windows:
	            for sp in subplots:
	                # get all trials that match this [shape, location]
	                trials = getAllTrialNumsForShapeAndLocInSession(sn, D, DS, shape, loc)
	                
	                # make plots
	                fig, axes = plotSkeletonsForTrialList(sn, trials, window[0], window[1], subplot=sp)
	                
	                # save plots
	                title = "/home/kgg/Desktop/" + str(date) + "_sn" + str(session) + "_" + animal + "_eyeplots/" + shape + '_loc' + str(loc) + '_EVENTS-' + window[0] + '-' + window[1] + '_SUBPLOT-' + str(sp) + ".png"
	                plt.savefig(title)
	                
	                # close plots so they do not show (saves memory...)
	                plt.close(fig)

###########################################################
### PART 2: Sequence-looking for primsingrid experiments###
###########################################################

# return a dict with {loc: #fixations_on_loc},...for all locs
# also returns FIRST fixation
def getNumFixationsAndFirstOnEachLocForTrial(sn, trial, start_event, end_event, keep_fixations_off_shape=False):
    # get all fixations
    fixations = getFixationsMatchedToTaskStrokes(sn, trial, start_event, end_event, keep_fixations_off_shape)
    # get all locations
    all_locs = getAllLocsForSession(sn, D, DS)
    
    # make dictionary using locations as keys, set all values as 0
    loc_freq = dict.fromkeys(all_locs, 0)
    
    # increment dictionary value if fixation is on loc
    for f in fixations:
        f_loc = f.task_stroke.token['gridloc']
        loc_freq[f_loc] = loc_freq[f_loc]+1
    return loc_freq, fixations[0].task_stroke.token['gridloc']

# get loc of first beh stroke for trial
def getFirstBehStrokeLocForTrial(sn, trial):
    bs = getAllBehStrokes(sn, trial)
    return bs[0].token['gridloc'], bs

# get sequence of beh stroke locs for trial
def getBehStrokeLocSequenceForTrial(sn, trial):
    bs = getAllBehStrokes(sn, trial)
    # get the loc and shape of each beh stroke
    sequence = [(strk.token["gridloc"], strk.token["shape"]) for strk in bs]
    return sequence




