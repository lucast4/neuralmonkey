#!/bin/bash -e
#
###################
#animal=Diego
#question=CHAR_BASE_trial
#ver=trial
#datelist=(231201 231205 231211)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 60m
#
#################
#animal=Diego
#question=PIG_BASE_trial
#ver=trial
#datelist=(230628 230629 230630)
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 60m
#
###################
#animal=Pancho
#question=CHAR_BASE_trial
#ver=trial
#datelist=(230119 230122 230126)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 60m
#
#################
#animal=Pancho
#question=PIG_BASE_trial
#ver=trial
##datelist=(230615 230615 230620 230621 230622)
#datelist=(230622 230623 230626)
#
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 60m


#############################
############################# RULE_STROKE (SINGLE RULE)
#animal=Diego
#question=RULE_BASE_stroke
#ver=stroke
##datelist=(230724 230726 230817 230913 230730 231118)
#datelist=(230724 230726 230817 230913 230730 231118)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 30s
#done
#sleep 30m
#
#animal=Pancho
#question=RULE_BASE_stroke
#ver=stroke
#datelist=(220901 220908 220909 230810 230826)
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


############################
############################ SHAPES VS. DIR (SWITCHING)
animal=Diego
question=RULESW_BASE_stroke
ver=stroke
#datelist=(230917 230823 230804 230809 230825 230813 230827 230919) # AnBmCk vs. DIR
datelist=(230705 230703 230711 230713 230719) # ABC vs. DIR
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 45m

#animal=Pancho
#question=RULESW_BASE_stroke
#ver=stroke
##datelist=(221020 230905 230910 230914 230919)
#datelist=(221023 230910 230914 230919)
##datelist=(230912 230914) # Ones that were missed
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



