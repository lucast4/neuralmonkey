#!/bin/bash -e

#################
animal=Diego

datelist=(240517 240521 240523 240730) # Just switching
which_plots=switching_dist

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 1m

#################
animal=Pancho

datelist=(240516 240521 240524) # Just switching
which_plots=switching_dist

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 10m


# #################
# animal=Diego

# datelist=(240517 240521 240523 240730) # Just switching
# which_plots=switching_ss

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
#   sleep 30s
# done
# sleep 1m

# #################
# animal=Pancho

# datelist=(240516 240521 240524) # Just switching
# which_plots=switching_ss

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
#   sleep 30s
# done
# sleep 10m


