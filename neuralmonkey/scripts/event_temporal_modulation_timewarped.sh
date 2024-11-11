#!/bin/bash -e

animal=$1

if [[ $animal == Diego ]]; then
  # datelist=(230615 230630) # Chars, those left over (added to previous, this gets all)
  datelist=(230619 231121 240809) #

elif [[ $animal == Pancho ]]; then
  # datelist=(220715 220603) # Chars, those left over (added to previous, this gets all)
  datelist=(220918 221220 240508) # Chars, those left over (added to previous, this gets all)
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

######
for date1 in "${datelist[@]}"
do
  logfile="../logs/event_temporal_modulation_timewarped-${date1}_${animal}.txt"
  touch ${logfile}  
  echo ${logfile}
  python ../analyses/event_temporal_modulation_timewarped.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 5s
done
# sleep 1m