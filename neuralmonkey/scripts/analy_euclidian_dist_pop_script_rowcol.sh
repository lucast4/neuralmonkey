#!/bin/bash -e

##################
animal=Pancho
question=RULE_ROWCOL_STROKE
which_level=stroke

#datelist=(230306 230307 230308 230309 230310 230320 230328) # all
datelist=(230309 230310 230320 230328) # subset
#datelist=(230320) # one day, testing.

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 30m
