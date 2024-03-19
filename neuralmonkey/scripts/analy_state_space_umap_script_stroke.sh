#!/bin/bash -e

################
#animal=Pancho
#question=PIG_BASE_stroke
#ver=stroke
##datelist=(230615 230615 230620 230621 230622)
#datelist=(230622 230626) # Best ones for online seqeunce
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

##################
animal=Pancho
question=CHAR_BASE_stroke
ver=stroke
datelist=(230119 230122 230126)
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 10s
done
sleep 30m


#################
#animal=Pancho
#question=PIG_BASE_trial
#ver=trial
#datelist=(230615 230615 230620 230621 230622)
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
#datelist=(230126)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 10m


################
#animal=Diego
#question=PIG_BASE_stroke
#ver=stroke
##datelist=(230624 230625 230626 230627 230629 230630)
#datelist=(230629 230630) # Best ones for online seqeunce
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

##################
animal=Diego
question=CHAR_BASE_stroke
ver=stroke
#datelist=(231201 231204 231219)
datelist=(231201 231205 231211)
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 10s
done
sleep 30m


#################
#animal=Diego
#question=PIG_BASE_trial
#ver=trial
#datelist=(230624 230625 230626 230627 230629)
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
#animal=Diego
#question=CHAR_BASE_trial
#ver=trial
#datelist=(231201 231204 231219)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
#  sleep 10s
#done
#sleep 30m
