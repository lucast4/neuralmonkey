#!/bin/bash -e


############################ TRIAL BY TRIAL
#############################
#animal=Diego
#question=RULESW_BASE_stroke
#ver=stroke
#datelist=(231017 231024 231019 231020)
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 1m
#done
#sleep 20m
#
#animal=Pancho
#question=RULESW_BASE_stroke
#ver=stroke
#datelist=(221015 221117 221120 221121 221119)
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 1m
#done
##sleep 10m

############################ BLOCK SWITCHING
############################
animal=Pancho
question=RULESW_BASE_stroke
ver=stroke
datelist=(220812 220816 220814 221106 221107 221113 221114)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
#sleep 10m

