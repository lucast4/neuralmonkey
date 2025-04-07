#!/bin/bash -e

combine=0

# # ###################
#  animal=Diego
#  question=RULE_ANBMCK_STROKE
#  which_level=stroke
#  # datelist=(230724 230726 230730 230817 230913 231116 231118) # ALL
#  #datelist=(230817) # Testing, 4/9
#  #datelist=(230724 230817 231116) # testing 4/10
#  #datelist=(230913) # testing 4/10
# # datelist=(230724 230726 230817 230913 231116 231118) # ignore 230730, not much vairaiton.
# # datelist=(230724 230726 230817 230913 231116 231118) # (most)
# #  datelist=(230724 230726 230817 230913 231116 231118) # (most)
# #  datelist=(230724 230726 230730 230817 230913 230915 231116 231118) # ALL
#  datelist=(230730 230914 230816) # Not yet done -- filling in gaps in spreadsheet

#  # To rerun entirely
# # datelist=()
#  # To plot
# # datelist=(231116 231118)

#  for date1 in "${datelist[@]}"
#  do
#    logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
#    touch ${logfile}
#    echo ${logfile}
#    python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
# #   sleep 2m
#    sleep 1m
#  done
# #  sleep 90m
# #  sleep 90m
# #sleep 2m


animal=Pancho
question=RULE_ANBMCK_STROKE
which_level=stroke
#datelist=(220902 220906 220907 220908 220909) # All with 2 shape sets
#datelist=(230811 230829 231114) # A few others
#datelist=(220909) # Testing 4/10
#  datelist=(220906 220907 220908 220909 230811 230829) # Most
# datelist=(220908 220909 230811 230829) # Missed
# datelist=(220902 230810 230826 230824) # Missed
# datelist=(231114 231116) # Missed

# datelist=(220902 220906 220907 220908 220909 230826 230824 230810 230811 230829 231114 231116) # ALL
datelist=(220831 220901 230810) # ALL

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} ${combine} 2>&1 | tee ${logfile} &
  sleep 1m
done
# sleep 60m
#  sleep 60m
