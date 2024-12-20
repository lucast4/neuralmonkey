#!/bin/bash -e

animal=$1
get_all_events=0

if [[ $animal == Diego ]]; then
  # datelist=(240625 240808 240809) # Syntax TI
  # question=RULE_ANBMCK_STROKE
  
  # datelist=(230730 230914 230816) # AnBmCk (not yet done)
  # question=RULE_ANBMCK_STROKE
  # combine=1

  ### SP vs. Char,
  # # datelist=(231130 231205 231211 231122 231128 231129 231201 231213 231204) # Chars
  # # datelist=(231120 231121 231206 231207 231218 231219 231220) # Chars, those left over (added to previous, this gets all)
  # datelist=(231120 231206 231207 231218 231220) # Chars, those left over (added to previous, this gets all)
  # question=CHAR_BASE_stroke
  # combine=1

  # ### Events modulation figures. Extract trial version, and include all events (e.g., go cue)
  # datelist=(230630) # Chars, those left over (added to previous, this gets all)
  # question=PIG_BASE_trial
  # combine=1 
  # get_all_events=1

  ### SINGLE PRIMS (SP)
  # datelist=(230614 230615 230618 230619 240508 240509 240510 240513 240530) # ALL (location and size) [DONE]
  # question=SP_BASE_trial
  # combine=1

  ### SP Psycho (switching)
  # datelist=(240517 240521 240523 240730) # ALL
  datelist=(240523) # failed one
  question=SP_psycho_trial
  combine=1

elif [[ $animal == Pancho ]]; then
  # # datelist=(220831 220901 220902 230810 230826 230824 231114 231116 230923 230921 230920 231019 231020) 
  # # datelist=(231114 231116 230921 230920) # Getting missing ones (8/2024)
  # datelist=(240619 240808 240809) # Syntax TI
  # combine=0
  # question=RULE_ANBMCK_STROKE

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

  ## SINGLE PRIMS (SP)
  # datelist=(220606 220608 220715 220716 220724 220918 240508 240530) # MANY (location and size). Picked good ones.
  # datelist=(220606 220716 220724 220918 240508 240530) # MANY (location and size). Picked good ones. [DONE]
  # datelist=(240515) # 
  # datelist=(220606 220717 240510 240530) # This completes extraction for all size dates.
  # datelist=(221218) # 
  datelist=(220715 220716 220717 220724 240530) # Rerunning the final dates, to get PMvl
  question=SP_BASE_trial
  combine=1

  ### SP Psycho (switching)
  # datelist=(240516 240521 240524) # ALL
  # question=SP_psycho_trial
  # combine=1

elif [[ $animal == Pancho_sp_chars ]]; then
  # HACK, just to get a few dates with PMv_l, 

  # --- SP/Chars, Getting "combined", just the good ones.
  datelist=(220614 220616 220621 220622 220624 220627 220618 220628 220630) # Additional ones
  combine=1
  question=CHAR_BASE_stroke
  animal=Pancho

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
  python analy_dfallpa_extract.py ${animal} ${date1} ${question} ${combine} ${get_all_events} 2>&1 | tee ${logfile} &
  sleep 1m
done
# sleep 1m