#!/bin/bash -e

##################
animal=Diego
question=RULESW_BASE_stroke
which_level=stroke
datelist=(230823 230804 230827 230919)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 20s
done
#sleep 60m
