#!/bin/bash -e


#################
animal=Diego

# # datelist=(240515 240517 240521 240523 240730 240731 240801 240802) # All(?)
datelist=(240515 240517 240523 240731 240801 240802) # Just smooth morphs
# datelist=(240515 240523) # redo failures

which_plots=smooth

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SMOOTH-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 1m
done
# sleep 1m

#################
animal=Pancho

# # datelist=(240516 240521 240524 240801 240802) # All?
datelist=(240516 240521 240524 240801 240802) # Just smooth morphs
# datelist=(240524) # redo failures

which_plots=smooth

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SMOOTH-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 1m
done
# sleep 10m


