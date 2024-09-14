#!/bin/bash -e


#################
animal=Diego

datelist=(240515 240517 240521 240523 240730) # Just switching
# datelist=(240515 240523) # redo failures
which_plots=switching

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 1m

#################
animal=Pancho

datelist=(240516 240521 240524) # Just switching
# datelist=(240524) # redo failures
which_plots=switching

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 10m


