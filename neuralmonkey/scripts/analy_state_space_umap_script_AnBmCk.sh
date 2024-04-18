#!/bin/bash -e


############################
############################ RULE_STROKE (SINGLE RULE)
animal=Diego
question=RULE_BASE_stroke
ver=stroke
#datelist=(230724 230726 230817 230913 230730 231116 231118)
#datelist=(231116 231118) # days missed
#datelist=(230730) # sterotped (not good)
datelist=(230724 230817 230913 231118) # days missed + quicker
#datelist=(230724 230817) # quicker
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 45m

#animal=Pancho
#question=RULE_BASE_stroke
#ver=stroke
##datelist=(220901 220908 220909 230810 230826)
##datelist=(220901 220909 230810 230826) # quicker
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 1m
#done
#sleep 60m
##
