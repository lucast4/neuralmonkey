#!/bin/bash -e

animal=$1

if [[ $animal == Diego ]]; then
  # datelist=(240625 240808 240809) # Syntax TI
  # question=RULE_ANBMCK_STROKE
  
  # datelist=(230730 230914 230816) # AnBmCk (not yet done)
  # question=RULE_ANBMCK_STROKE
  # combine=1

  datelist=(231130 231205 231211 231122 231128 231129 231201 231213 231204) # Chars
  question=CHAR_BASE_stroke
  combine=1

elif [[ $animal == Pancho ]]; then
  # # datelist=(220831 220901 220902 230810 230826 230824 231114 231116 230923 230921 230920 231019 231020) 
  # # datelist=(231114 231116 230921 230920) # Getting missing ones (8/2024)
  # datelist=(240619 240808 240809) # Syntax TI
  # combine=0
  # question=RULE_ANBMCK_STROKE

  # datelist=(230122 230125 230126 230127 230112 230117 230118 230119 230120) # Chars [set 1]
  datelist=(220531 220602 220603 220614 220616 220618 220626 220627 220628 220630) # Chars [set 2] -- Getting more dates, so that combined with previous, is all.

  combine=0
  question=CHAR_BASE_stroke
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
  python analy_dfallpa_extract.py ${animal} ${date1} ${question} ${combine} 2>&1 | tee ${logfile} &
  sleep 2m
done
# sleep 1m