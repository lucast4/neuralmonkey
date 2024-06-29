#!/bin/bash -e

###################
animal=Pancho
question=RULESW_ANBMCK_ABN_STROKE
which_level=stroke
datelist=(220829 220830) # All
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 30s
done
#sleep 10m
