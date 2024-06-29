#!/bin/bash -e

###################
animal=Diego
question=SP_BASE_stroke
which_level=stroke
datelist=(230614 230615 230618 230619)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 30m


###################
animal=Pancho
question=SP_BASE_stroke
which_level=stroke
datelist=(220716 220717 220715 220918)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 30m
