#!/bin/bash -e


############################
############################ RULE_STROKE (SINGLE RULE)
animal=Pancho
question=RULE_BASE_stroke
ver=stroke
#datelist=(230306 230307 230308 230309 230310 230320 230328) # all
datelist=(230310 230320 230328) # subset
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 1m
