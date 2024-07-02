#!/bin/bash -e


##################
animal=Diego
question=SP_BASE_trial
# datelist=(230603 230613 230614 230615 230616 230617 230618 230619 230621) # Further subset, for rducing workload on copuater
datelist=(230615 230616 230618 230619 240508 240509 240510 240530) # Further subset, for rducing workload on copuater
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 180m

##################
animal=Pancho
question=SP_BASE_trial
datelist=(220715 220716 240508 240509 220724 240510 240530) # Further subset, for rducing workload on copuater
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 2m
done
