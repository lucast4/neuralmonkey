#!/bin/bash -e

#################
animal=Diego

datelist=(240517 240521 240523 240730) # Just switching
# datelist=(240523) # Just switching
which_plots=switching_dist

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 5m

#################
animal=Pancho

datelist=(240516 240521 240524) # Just switching
# datelist=(240524) # Redo (KS extracted)
which_plots=switching_dist

for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_decode_moment_psychometric-SWITCHING-${animal}_${date1}_${which_plots}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_decode_moment_psychometric.py ${animal} ${date1} ${which_plots} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 30s


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
# # datelist=(240516) # Redo (KS extracted)
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


