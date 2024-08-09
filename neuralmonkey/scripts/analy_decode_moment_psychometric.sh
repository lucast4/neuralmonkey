#!/bin/bash -e

#################
animal=Diego
datelist=(240515 240517 240521 240523 240730 240731 240801 240802)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 60m


#################
animal=Pancho
datelist=(240516 240521 240524 240801 240802)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 60m


