#!/bin/bash -e


############################ BLOCK SWITCHING
############################
animal=Pancho
question=RULESW_BASE_stroke
ver=stroke
datelist=(221111 221112)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
#sleep 10m
