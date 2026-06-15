#!/bin/bash -e

##################
animal=Diego

question=RULE_ANBMCK_STROKE
# datelist=(230726 230815 230816 230817 230913 230915 231118) # Good subset of dates, just for testing
datelist=(230726 230815 230816 230817 230913 230915 231118) 

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_smfr_units-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_smfr_units.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done
# sleep 3h
sleep 1m

##################
animal=Pancho

question=RULE_ANBMCK_STROKE
datelist=(220909 230824 230829 231114 231116 250322) 
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_smfr_units-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_smfr_units.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done
