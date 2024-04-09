#!/bin/bash -e


############################
############################ SHAPES VS. DIR (SWITCHING)
animal=Diego
question=RULEVSCOL_BASE_stroke
ver=stroke
#datelist=(230910 230911 230907 230912 230927 231001) # ALL
#datelist=(230910 230907 230912 231001) # subset, just to do during day
#datelist=(230910 230912) # quick, good
#datelist=(230912) # quick, good (subset)
datelist=(230910 230911 230907 230927 231001) # just those missed..

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 30m
#
animal=Pancho
question=RULEVSCOL_BASE_stroke
ver=stroke
datelist=(230928 230929)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 5m
