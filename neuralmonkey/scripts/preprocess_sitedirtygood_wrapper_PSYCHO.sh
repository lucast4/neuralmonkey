#!/bin/bash -e

###################
animal=Diego
datelist=(240515 240517 240731 240801 240802)

for date1 in "${datelist[@]}"
do
  logfile="../logs/preprocess_sitedirtygood_wrapper-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python preprocess_sitedirtygood_wrapper.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 20m

###################
animal=Pancho
datelist=(240516 240521 240524 240801 240802)

for date1 in "${datelist[@]}"
do
  logfile="../logs/preprocess_sitedirtygood_wrapper-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python preprocess_sitedirtygood_wrapper.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 20m

