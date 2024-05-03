#!/bin/bash -e

###################
animal=Diego
question=PIG_BASE_stroke
which_level=stroke
# datelist=(230628 230629 230630)
datelist=(230628 230630)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 30m
# sleep 2m

###################
animal=Pancho
question=PIG_BASE_stroke
which_level=stroke
datelist=(230623 230626)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 30m
# sleep 2m
