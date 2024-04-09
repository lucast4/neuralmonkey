#!/bin/bash -e

########################### AnBm vs. (AB)n
animal=Pancho
question=RULESW_BASE_stroke
ver=stroke
datelist=(220829 220830) # All
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 30s
done
#sleep 45m
