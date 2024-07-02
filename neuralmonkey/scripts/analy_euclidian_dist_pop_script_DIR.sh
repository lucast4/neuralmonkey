#!/bin/bash -e

##################
animal=Diego
question=RULE_DIR_STROKE
which_level=stroke
datelist=(230929)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 1m
done

animal=Pancho
question=RULE_DIR_STROKE
which_level=stroke
datelist=(231017)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 30m
