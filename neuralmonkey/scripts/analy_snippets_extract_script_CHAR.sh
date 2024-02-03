#!/bin/bash -e

# sleep 1h
animal=$1
#analy=$2

if [[ $animal == Diego ]]; then
  datelist=(231201 231204 231219)
elif [[ $animal == Pancho ]]; then
  datelist=(230120 230122 230125 230126 230127)
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

which_level_list=(trial stroke stroke_off)
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
    taskset --cpu-list 0,1,2,3,4,5,6 python analy_snippets_extract.py ${animal} ${date1} ${which_level} 2>&1 | tee ${logfile} &
    sleep 5s
  done
  sleep 5m
done