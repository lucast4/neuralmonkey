#!/bin/bash -e

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
question=CHAR_BASE_stroke
which_level=stroke
datelist=(230122 230125 230126 230127 230112 230117 230118 230119 230120) # ALL
combine=0
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
