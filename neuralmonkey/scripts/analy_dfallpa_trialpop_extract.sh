#!/bin/bash -e

animal=$1

if [[ $animal == Diego ]]; then
  # datelist=(230615 230630) # Chars, those left over (added to previous, this gets all)
  # datelist=(230619 231121 240809) #
  # datelist=(230618 230619) # Shape vs. size
  datelist=(230618) # Shape vs. size

elif [[ $animal == Pancho ]]; then
  # datelist=(220715 220603) # Chars, those left over (added to previous, this gets all)
  # datelist=(220918 221220 240508) # Chars, those left over (added to previous, this gets all)
  # datelist=(220716 220724 220712 220719) # Chars, those left over (added to previous, this gets all)
  datelist=(220716 220717 240530) # Shape vs. size
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

######
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_dfallpa_trialpop_extract-${date1}_${animal}.txt"
  touch ${logfile}  
  echo ${logfile}
  # taskset --cpu-list 0,1,2,3,4,5,6 python analy_snippets_extract.py ${animal} ${date1} ${question} ${combine} 2>&1 | tee ${logfile} &
  python analy_dfallpa_trialpop_extract.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 5s
done
# sleep 1m