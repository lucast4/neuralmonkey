#!/bin/bash -e

###################
animal=Pancho
datelist=(220606 220608 220610 220715 220716 220717 220724 220918 221218 240508 240509 240510 240515 240530) # all that include location (7/24/24)
# datelist=(220717 220610 220724 220918 240510 240515 240530) # round 1
# datelist=(220606 220608 220715 220716 221218 240508 240509) # round 2
# datelist=(220715 220716 221218 240508 240509) # including scalar state space

for date1 in "${datelist[@]}"
do
  logfile="../logs/preprocess_sitedirtygood_wrapper-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python preprocess_sitedirtygood_wrapper.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 8m
done
sleep 20m

###################
animal=Diego
datelist=(230614 230615 230618 230619 240508 240509 240513 240510 240530) # all that include location (7/24/24)
# datelist=(240510 240513 240530) # round 1
# datelist=(230614 230615 230618 230619 240508 240509) # round 2
# datelist=(230615 240508 240509 240510 240530) # including scalar state space

for date1 in "${datelist[@]}"
do
  logfile="../logs/preprocess_sitedirtygood_wrapper-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python preprocess_sitedirtygood_wrapper.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 8m
done
sleep 20m

