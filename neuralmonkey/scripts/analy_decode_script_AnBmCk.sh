#!/bin/bash -e

#################
animal=Diego
question=RULE_BASE_stroke
ver=stroke
datelist=(230817)

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_script-${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_script.py ${animal} ${date1} 0 ${question} ${ver} 0100 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 60m

