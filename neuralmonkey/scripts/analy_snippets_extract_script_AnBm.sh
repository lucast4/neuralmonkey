#!/bin/bash -e

# sleep 1h
animal=$1
#analy=$2

if [[ $animal == Diego ]]; then
  datelist=(230817)
elif [[ $animal == Pancho ]]; then
  datelist=()
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

#which_level_list=(trial stroke stroke_off)
which_level_list=(trial stroke)
FORCE_EXTRACT=0

#
#if [[ $analy == PIG ]]; then
#  which_level_list=(stroke stroke_off)
#else
#  echo $analy
#  echo "Error! Inputed non-existing analy" 1>&2
#  exit 1
#fi

echo "This animal: $animal"
#echo "This analy: $analy"
echo "These dates: ${datelist[*]}"
echo "These which_level_list: ${which_level_list[*]}"

######
for which_level in "${which_level_list[@]}"
do
  for date1 in "${datelist[@]}"
  do
    logfile="../logs/log_analy_snippets_extract_${which_level}_${date1}_${animal}.txt"
    touch ${logfile}
    echo ${logfile}
    taskset --cpu-list 0,1,2,3,4,5,6 python analy_snippets_extract.py ${animal} ${date1} ${which_level} ${FORCE_EXTRACT} 2>&1 | tee ${logfile} &
    sleep 5s
  done
  sleep 2m
done