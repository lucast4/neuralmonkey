#!/bin/bash -e

# sleep 1h
animal=$1
#analy=$2

if [[ $animal == Diego ]]; then
#  datelist=(230630 230628) #
  datelist=(230624 230625 230626 230627 230629) # extra (3/2/24)
  datelist=(230629 230630) # best days for online decode sequential context (also 230627?)
elif [[ $animal == Pancho ]]; then
#  datelist=(230623 230666)
  datelist=(230615 230615 230620 230621 230622) # extra (3/2/24)
  datelist=(230622 230626) # best days for online decode sequential context (also 230623?)
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
    sleep 10s
  done
  sleep 30m
done