from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper

# LIST_DATE = [
# 	"221019", "221020", "221021", "221023", "221024", 
# 	"221029", "221031", "221102",
# 	"221113", "221114", 
# 	"221116", "221117", "221119", "221120", "221121", "221122", "221123", "221125"]
LIST_DATE = [
	"221125",
	"221113", "221114",
	"221031", "221102", 
	"221014", 
	"221001", "221002", 
	"220925", "220926", "220928", "220929", "220930", 
	"220825", "220826", "220827",
	"221119", "221120", "221121",
	# "221122", "221123", 
	# "221125"
	]
LIST_DATE = LIST_DATE[::-1] # flip, for 6b

MINIMAL_PLOTS = True
LIST_DATASET_PRUNE_SAME_BEH_ONLY = [False]
LIST_FIRST_STROKE_SAME = [False]
DEBUG=True
PREFIX_SAVE_BASE = "rulesgood"
for DATE in LIST_DATE:
	for DATASET_PRUNE_SAME_BEH_ONLY in LIST_DATASET_PRUNE_SAME_BEH_ONLY:
		for FIRST_STROKE_SAME in LIST_FIRST_STROKE_SAME:
			# if DATE == "221125" and DATASET_PRUNE_SAME_BEH_ONLY==False:
			# 	# ALREADY DONE
			# 	continue
			animal = "Pancho"
			dataset_beh_expt = None
			# DATASET_PRUNE_SAME_BEH_ONLY = True

			# %matplotlib inline
			# to help debug if times are misaligned.

			# MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)
			MS = load_mult_session_helper(DATE, animal)
			
			min_trials=150
			MS.prune_remove_sessions_too_few_trials(min_trials)

			# [OPTIONAL] import dataset
			# for sn in MS.SessionsList:
			#     sn.datasetbeh_load_helper(dataset_beh_expt)
			for sn in MS.SessionsList:
				sn.datasetbeh_load_helper(dataset_beh_expt)

			for sn in MS.SessionsList:
				D = sn.Datasetbeh

				# Re-Define taskgroups even if they are not probes.
				# from pythonlib.dataset.dataset_preprocess.probes import compute_features_each_probe, taskgroups_assign_each_probe
				# taskgroups_assign_each_probe(D, False)
				D.taskgroup_reassign_ignoring_whether_is_probe()


			from neuralmonkey.classes.snippets import datasetstrokes_extract
			from neuralmonkey.analyses.site_anova import _dataset_extract_prune_rulesw, _dataset_extract_prune_sequence, params_database_extract
			import os
			import numpy as np
			import seaborn as sns
			from neuralmonkey.classes.snippets import Snippets

			QUESTION = "rulesw"
			EVENTS_SIMPLE = False
			REMOVE_BASELINE_EPOCHS=True
			n_min_trials_in_each_epoch = 1

			PARAMS = params_database_extract(MS, QUESTION, EVENTS_SIMPLE, 
					DATASET_PRUNE_SAME_BEH_ONLY, REMOVE_BASELINE_EPOCHS, n_min_trials_in_each_epoch,
											 FIRST_STROKE_SAME, DEBUG, MINIMAL_PLOTS)

			SESSIONS = PARAMS["SESSIONS"]
			LIST_PLOTS = PARAMS["LIST_PLOTS"]
			THINGS_TO_COMPUTE = PARAMS["THINGS_TO_COMPUTE"]
			list_events = PARAMS["list_events"]
			list_pre_dur = PARAMS["list_pre_dur"]
			list_post_dur = PARAMS["list_post_dur"]
			list_features_get_conjunction = PARAMS["list_features_get_conjunction"]
			list_features_extraction = PARAMS["list_features_extraction"]
			list_possible_features_datstrokes = PARAMS["list_possible_features_datstrokes"]


			import os
			if FIRST_STROKE_SAME:
			    PREFIX_SAVE = f"{PREFIX_SAVE_BASE}_samefirststroke"
			elif DATASET_PRUNE_SAME_BEH_ONLY:
			    PREFIX_SAVE = f"{PREFIX_SAVE_BASE}_samebeh"
			else:
			    PREFIX_SAVE = f"{PREFIX_SAVE_BASE}_all"
			    
			# SAVEDIR = f"/data2/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"
			if PREFIX_SAVE is None:
			    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"
			else:
			    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}_{PREFIX_SAVE}"
			    
			os.makedirs(SAVEDIR, exist_ok=True)
			print(SAVEDIR)

			prune_feature_levels_min_n_trials=n_min_trials_in_each_epoch
			for sess in SESSIONS:
				sn = MS.SessionsList[sess]

				# pass in already-pruned/preprocessed dataset?
				if QUESTION=="rulesw":
					dataset_pruned_for_trial_analysis = _dataset_extract_prune_rulesw(sn, DATASET_PRUNE_SAME_BEH_ONLY, 
						n_min_trials_in_each_epoch=n_min_trials_in_each_epoch, remove_baseline_epochs=REMOVE_BASELINE_EPOCHS,
						first_stroke_same_only=FIRST_STROKE_SAME)
				elif QUESTION=="sequence":
					dataset_pruned_for_trial_analysis = _dataset_extract_prune_sequence(sn, n_strok_max=2)
				else:
					dataset_pruned_for_trial_analysis = None
				
				if DEBUG and dataset_pruned_for_trial_analysis is None:
					dataset_pruned_for_trial_analysis = sn.Datasetbeh.copy()
					dataset_pruned_for_trial_analysis.Dat = dataset_pruned_for_trial_analysis.Dat[-25:]
				elif DEBUG:
					dataset_pruned_for_trial_analysis.Dat = dataset_pruned_for_trial_analysis.Dat[-25:]
				
				SAVEDIR_SCALAR = f'{SAVEDIR}/sess_{sess}/scalar_plots'
				os.makedirs(SAVEDIR_SCALAR, exist_ok=True)

				##### 1) First automatically figure out what features to use
				# - extract datstrokes, and check what fetures it has
				if QUESTION in ["motor", "stroke"]:
					strokes_only_keep_single=True
					tasks_only_keep_these=["prims_single"]
					DS = datasetstrokes_extract(sn.Datasetbeh, strokes_only_keep_single,
												tasks_only_keep_these,
												prune_feature_levels_min_n_trials, list_possible_features)
					if len(DS.Dat)<50:
						continue

					# check which features exist
					list_features_extraction = []
					for feat in list_possible_features:
						levels = DS.Dat[feat].unique().tolist()
						if len(levels)>1:
							# keep
							list_features_extraction.append(feat)
					print("=== USING THESE FEATURES:", list_features_extraction)
					assert len(list_features_extraction)>0

				elif QUESTION in ["sequence", "rulesw"]:
					# DOesnt need DS
					pass 
				else:
					print(QUESTION)
					assert False
				

				##### 2) Do everything
				if QUESTION in ["motor", "stroke"]:
					list_features_get_conjunction = list_features_extraction
					strokes_only_keep_single = True
					which_level = "stroke"
				elif QUESTION in ["sequence", "rulesw"]:
					strokes_only_keep_single = False
					which_level = "trial"
				else:
					assert False
					
				print("Extracvting snips..")
				SP = Snippets(SN=sn, which_level=which_level, list_events=list_events, 
							list_features_extraction=list_features_extraction, list_features_get_conjunction=list_features_get_conjunction, 
							list_pre_dur=list_pre_dur, list_post_dur=list_post_dur,
							strokes_only_keep_single=strokes_only_keep_single,
							  prune_feature_levels_min_n_trials=prune_feature_levels_min_n_trials,
							  dataset_pruned_for_trial_analysis = dataset_pruned_for_trial_analysis
							 )
				

				if dataset_pruned_for_trial_analysis is not None:
					# Sanity check: correct subset of trials
					tc1 = sorted(SP.DfScalar["trialcode"].unique().tolist())
					tc2 = sorted(dataset_pruned_for_trial_analysis.Dat["trialcode"].unique().tolist())
					assert tc1 == tc2, "Pruned dataset did not work correctly in Snippets"
					
				########################################## PLOTS
				# Compute summary stats
				RES_ALL_CHANS = SP.modulation_compute_each_chan(things_to_compute=THINGS_TO_COMPUTE)
				OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)
				
				### PLOT max mod each site.
				list_max = []
				list_site = []
				for i in range(len(RES_ALL_CHANS)):
					val = np.max(RES_ALL_CHANS[i]["RES"]["modulation_across_events"]["epoch"])
					list_max.append(val)
					list_site.append(RES_ALL_CHANS[i]["chan"])
				fig,ax = plt.subplots(figsize = (4,15))
				ax.plot(list_max, list_site, 'ok')
				ax.set_ylabel('site')
				ax.set_xlabel('max modulation across events')
				plt.grid(True, "major")
				fig.savefig(f"{SAVEDIR_SCALAR}/sites_max_modulation.pdf")

				# Plot and save
				if False:
					# Dont need to do this, since rules have character as an explicit variable
					if QUESTION=="rulesw":
						# because interested in whether activity encodes the rule or the 
						# action plan (character)
						list_plots = list_plots + ["eachsite_smfr_splitby_character", "eachsite_raster_splitby_character"]
				SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, list_plots=LIST_PLOTS)
