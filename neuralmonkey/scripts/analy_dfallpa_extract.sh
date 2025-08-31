#!/bin/bash -e

animal=$1
get_all_events=0
use_spike_counts=0

if [[ $animal == Diego ]]; then
  # datelist=(240625 240808 240809) # Syntax TI
  # question=RULE_ANBMCK_STROKE
  
  # ## AnBmCk and AnBmCk(Two shape sets)
  # datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118 240822 240827 250319 250321) # AnBmCk, ALL
  # datelist=(230817 230914) # AnBmCk, ALL
  # datelist=(230728 230817) # NEED TO FIX

  # # # datelist=(250319 250321) # added 4/4/25
  # # datelist=(250319) # added 4/4/25
  # datelist=(230914 231116) # AnBmCk, 7/31/25, best days, for testing
  datelist=(240822) # AnBmCk, 7/31/25, best days, for testing
  question=RULE_ANBMCK_STROKE
  combine=0

  # ## AnBmCk -- getting also the single prim trials
  # # datelist=(230726 230913 231118 240827) # AnBmCk, 7/31/25, best days, for testing
  # datelist=(230723 230724 230727 230728 230730 230815 230816 230817 230914 230915 231116 240822 250319 250321) # AnBmCk, 7/31/25, best days, for testing
  # question=SP_BASE_stroke
  # combine=0

  # ### AnBmCk trial (all, for trial not stroke)
  # # datelist=(230728 231118 240822) # 
  # datelist=(230728 231118 240822 230723 230724 230726 230727 230730 230815 230816 230817 230913 230914 230915 231116 240827 250319 250321) # 
  # question=RULE_BASE_trial
  # combine=0

  ### SP vs. Char,
  # # datelist=(231130 231205 231211 231122 231128 231129 231201 231213 231204) # Chars
  # # datelist=(231120 231121 231206 231207 231218 231219 231220) # Chars, those left over (added to previous, this gets all)
  # datelist=(231120 231206 231207 231218 231220) # Chars, those left over (added to previous, this gets all)
  # question=CHAR_BASE_stroke
  # combine=1

  # # SP vs. Char,
  # # datelist=(231130 231205 231211 231122 231128 231129 231201 231213 231204) # Chars
  # # datelist=(231120 231121 231206 231207 231218 231219 231220) # Chars, those left over (added to previous, this gets all)
  # # datelist=(231120 231206 231207 231218 231220) # Chars, those left over (added to previous, this gets all)
  # datelist=(231120 231122 231128 231129 231201 231205 231206 231218 231220) # ALL (neural)
  # # datelist=(231206) # KS gotten
  # question=CHAR_BASE_stroke
  # # question=CHAR_BASE_trial
  # combine=1
  # # get_all_events=1

  # ### Events modulation figures. Extract trial version, and include all events (e.g., go cue)
  # datelist=(230630) # Chars, those left over (added to previous, this gets all)
  # question=PIG_BASE_trial
  # combine=1 
  # get_all_events=1

  # ## SINGLE PRIMS (SP)
  # datelist=(230614 230615 230618 230619 240508 240509 240510 240513 240530) # ALL (location and size) [DONE]
  # # question=SP_BASE_trial
  # question=SP_BASE_stroke
  # combine=1

  # ### SP Psycho (switching)
  # # datelist=(240517 240521 240523 240730) # ALL
  # datelist=(240523) # failed one
  # question=SP_psycho_trial
  # combine=1

  # PIG (saccade fixation)
  # datelist=(230615 230628 230630 240625) 
  # datelist=(230628 230630 240625) 
  # question=PIG_BASE_saccade_fix_on
  # combine=1
  
  # # PIG (trial)
  # datelist=(230628 240625) 
  # question=PIG_BASE_trial
  # combine=1 

elif [[ $animal == Diego_seqsup ]]; then
  ### AnBmCk vs. SEQSUP
  datelist=(230920 230921 230922 230924 230925 250320) # AnBmCk, ALL
  # datelist=(230921 230924 230925) # missed
  # datelist=(250320) # missed
  question=RULESW_ANY_SEQSUP_STROKE
  combine=0
  animal=Diego

elif [[ $animal == Pancho ]]; then
  
  # datelist=(240619 240808 240809) # Syntax TI

  ## AnBmCk and AnBmCk(Two shape sets)
  # datelist1=(230810 230811 230824 230826 230829 231114 231116 240830 220831 220901 220902 220906 220907 220908 220909 250321 250322) # AnBmCk, ALL
  # datelist1=(230811 230824 230826 230829 231114 231116) # AnBmCk, SUBSET
  # datelist1=(220831 220901) # NEED TO FIX BUG
  # datelist1=(250322) # added 4/4/25
  # datelist1=(230908 230909 231114 231116) # AnBmCk, Good ones, for testing, 7/31/25
  datelist1=(230810 230811 230826) # Missed ones
  datelist2=() # SUBSET
  datelist=(${datelist1[@]} ${datelist2[@]})
  echo ${datelist[@]}
  combine=0
  question=RULE_ANBMCK_STROKE

  # ## AnBmCk -- getting also the single prim trials
  # # datelist1=(230908 230909 231114 231116) # AnBmCk, Good ones, for testing, 7/31/25
  # datelist1=(230810 230811 230824 230826 230829 240830 220831 220901 220902 220906 220907 220908 220909 250321 250322) # AnBmCk, Good ones, for testing, 7/31/25
  # datelist2=() # SUBSET
  # datelist=(${datelist1[@]} ${datelist2[@]})
  # echo ${datelist[@]}
  # combine=0
  # question=SP_BASE_stroke

  # ### AnBmCk trial (all, for trial not stroke)
  # # datelist=(230810 230811 230824 230826 230829 231114 231116 240830 220831 220901 250321 250322 220902 220906 220907 220908 220909) # 
  # # datelist=(230810 230811 230824 230826 230829 231114 231116 240830 220831 220901 250321 250322) # 
  # # datelist=(220902 220906 220907 220908 220909) # 
  # datelist=(250322) # 
  # question=RULE_BASE_trial
  # combine=0

  # --- SP/Chars, all
  # datelist=(230122 230125 230126 230127 230112 230117 230118 230119 230120) # Chars [set 1]
  # datelist=(220531 220602 220603 220614 220616 220618 220626 220627 220628 220630) # Chars [set 2] -- Getting more dates, so that combined with previous, is all.
  # combine=0
  # question=CHAR_BASE_stroke

  # # --- SP/Chars, Getting "combined", just the good ones.
  # # datelist=(220618 220626 220628 220630 230119 230120 230126 230127) # Combining sets 1 and 2, to those that passed clean criteria (see spreadsheet)
  # datelist=(220614 220616 220621 220622 220624 220627 230112 230117 230118) # Additional ones
  # combine=1
  # question=CHAR_BASE_stroke

  # ### Events modulation figures. Extract trial version, and include all events (e.g., go cue)
  # datelist=(220715) # Chars, those left over (added to previous, this gets all)
  # question=SP_BASE_trial
  # combine=1 
  # get_all_events=1

  # ## SINGLE PRIMS (SP)
  # # datelist=(220606 220608 220715 220716 220724 220918 240508 240530) # MANY (location and size). Picked good ones.
  # # datelist=(220606 220716 220724 220918 240508 240530) # MANY (location and size). Picked good ones. [DONE]
  # # datelist=(240515) # 
  # # datelist=(220606 220717 240510 240530) # This completes extraction for all size dates.
  # # datelist=(221218) # 
  # datelist=(220715 220716 220717 220724 240530) # ALL DATES Rerunning the final dates, to get PMvl
  # datelist=(220716) # missed dates
  # # question=SP_BASE_trial
  # question=SP_BASE_stroke
  # combine=1

  ### SP Psycho (switching)
  # datelist=(240516 240521 240524) # ALL
  # question=SP_psycho_trial
  # combine=1

  # # PIG (trial)
  # # datelist=(230620 230622 230623 230626 240612 240618) # all
  # # datelist=(230622 230626 240612) # subset, just to be quicker
  # datelist=(240612 240618) # all
  # question=PIG_BASE_trial
  # combine=1 

elif [[ $animal == Pancho_pig_sacc_fix ]]; then
  # PIG (saccade fixation)

  # datelist=(230620 230622 230623 230626 240612 240618) # all
  # datelist=(230622 230626 240612) # subset, just to be quicker
  datelist=(240612 240618) # all
  question=PIG_BASE_saccade_fix_on
  combine=1
  animal=Pancho

elif [[ $animal == Pancho_seqsup ]]; then
  
  ### AnBmCk vs. SEQSUP
  datelist=(230920 230921 230923 231019 231020 240828 240829 250324 250325) # All
  # datelist=(240828 240829) # missed
  # datelist=(231020) # flipped cue-image onset... need to fix this, not working.
  # datelist=(250324 250325) # added 4/4/25
  combine=0
  animal=Pancho
  question=RULESW_ANY_SEQSUP_STROKE

elif [[ $animal == Pancho_sp_chars ]]; then
  # HACK, just to get a few dates with PMv_l, 

  # --- SP/Chars, Getting "combined", just the good ones.

  # Stroke
  # datelist=(220614 220616 220618 220621 220622 220624 220626 220627 220628 220630) # Additional ones
  # combine=1
  # question=CHAR_BASE_stroke
  # animal=Pancho

  # Trial
  # datelist=(220614 220616 220618 220621 220622 220624 220626 220627 220628 220630) # Additional ones
  # datelist=(220616 220621 220622 220624 220626 220627) # Additional ones
  datelist=(220614) # Additional ones
  combine=1
  question=CHAR_BASE_stroke
  animal=Pancho
  # question=CHAR_BASE_trial
  # get_all_events=0

elif [[ $animal == Pancho_switching ]]; then
  # HACK, just to get a few dates with PMv_l,   

  ## SP Psycho (switching)
  # datelist=(240516 240521 240524) # ALL
  datelist=(240516) # missed
  question=SP_psycho_trial
  combine=1
  animal=Pancho

elif [[ $animal == Pancho_not_combined ]]; then
  # HACK, just to get a few dates with PMv_l, 

  # --- SP/Chars, Getting "combined", just the good ones.
  datelist=(220618 220626 220628 220630 230119 230120 230126 230127) # Combining sets 1 and 2, to those that passed clean criteria (see spreadsheet)
  combine=0
  question=CHAR_BASE_stroke
  animal=Pancho

elif [[ $animal == Diego_AnBmCk_LFADS ]]; then
  # 8/18/25 - For testing LFADS

  # AnBmCk -- getting also the single prim trials
  # datelist=(230726 230913 231118 240827) # AnBmCk, 7/31/25, best days, for testing
  datelist=(230913) #
  question=RULE_ANBMCK_STROKE
  # question=SP_BASE_stroke
  combine=0
  use_spike_counts=1
  animal=Diego
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

######
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_dfallpa_extract-${question}_${date1}_${animal}_${combine}.txt"
  touch ${logfile}  
  echo ${logfile}
  # taskset --cpu-list 0,1,2,3,4,5,6 python analy_snippets_extract.py ${animal} ${date1} ${question} ${combine} 2>&1 | tee ${logfile} &
  python analy_dfallpa_extract.py ${animal} ${date1} ${question} ${combine} ${get_all_events} ${use_spike_counts} 2>&1 | tee ${logfile} &
  sleep 2m
done
# sleep 1m