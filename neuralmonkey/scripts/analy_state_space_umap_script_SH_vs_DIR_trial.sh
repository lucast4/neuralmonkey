#!/bin/bash -e


###########################
########################### SHAPES VS. DIR (TRIAL)
animal=Diego
question=RULESW_BASE_stroke
ver=stroke
#datelist=(230917 230823 230804 230809 230825 230813 230827 230919) # AnBmCk vs. DIR
#datelist=(230705 230703 230711 230713 230719) # ABC vs. DIR
datelist=(230823 230804 230919) # quick subset good
#datelist=(230823) # quick subset good (smaller)
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 60m

#animal=Pancho
#question=RULESW_BASE_stroke
#ver=stroke
##datelist=(221020 230905 230910 230914 230919)
datelist=(221023 230910 230912 230914 230919)
##datelist=(230912 230914) # quick subset
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 1m
#done
#sleep 10m
