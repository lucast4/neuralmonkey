#!/bin/bash -e

PLOT_DO=2

##################
animal=Diego
# datelist=(230726 230815 230816 230817 230913 230915 231118)
datelist=(230815 230816 230913)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_errors_smfr_pop-${animal}_${date1}_${PLOT_DO}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_errors_smfr_pop.py ${animal} ${date1} ${PLOT_DO} 2>&1 | tee ${logfile} &
  sleep 10s
done

# sleep 3h
sleep 5m

##################
animal=Pancho
# datelist=(220909 230824 230829 231114 231116)
datelist=(230824 230829 231114 231116)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_errors_smfr_pop-${animal}_${date1}_${PLOT_DO}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_errors_smfr_pop.py ${animal} ${date1} ${PLOT_DO} 2>&1 | tee ${logfile} &
  sleep 10s
done

# sleep 3h
# sleep 1m
