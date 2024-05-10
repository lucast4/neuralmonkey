#!/bin/bash -e

#################
animal=Diego
question=RULE_BASE_stroke
ver=stroke
datelist=(230928)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_script-${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_script.py ${animal} ${date1} 0 ${question} ${ver} 0100 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 60m

#animal=Pancho
#question=RULEVSCOL_BASE_stroke
#ver=stroke
#datelist=(231001) # all
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 2m
#done
#sleep 60m

