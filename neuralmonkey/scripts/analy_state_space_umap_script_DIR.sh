#!/bin/bash -e


###########################
animal=Diego
question=RULE_BASE_stroke
ver=stroke
datelist=(230929) # all
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 1m
#
animal=Pancho
question=RULE_BASE_stroke
ver=stroke
datelist=(231017) # all
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 20m
