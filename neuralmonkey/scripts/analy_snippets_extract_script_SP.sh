#!/bin/bash -e

# sleep 1h
animal=$1
#analy=$2

if [[ $animal == Diego ]]; then
#  datelist=(230603 230613 230614 230615 230616 230617 230618 230619 230621) # all
#  datelist=(230614 230615 230618 230619) # good ones, temp
  datelist=(230614 230618 230619) # good ones, temp
elif [[ $animal == Pancho ]]; then
#  datelist=(220715 220716 220717 220606 220608 220609 220610 220718 220719 220724 220918 221217 221218 221220 230103 230104) # the ones added 6/13/23, to get all days.
  datelist=(220715 220716 220610 220918) # the ones added 6/13/23, to get all days.
#  datelist=(220606 220608 220609 220610) # (shape, loc, size)
#  datelist=(230612 230613) # complexvar
#  datelist=(221220 230103 230104 220718 221217) # novel prims
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