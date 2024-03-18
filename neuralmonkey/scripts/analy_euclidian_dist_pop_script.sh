#!/bin/bash -e

##################
animal=Diego
#question=CHAR_BASE_trial
#datelist=(231201 231204 231219 231130 231205 231207 231211 231220)
question=PIG_BASE_stroke
which_level=stroke
#datelist=(230625 230626 230628 230627 230629 230630)
datelist=(230630)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 20s
done
#sleep 60m

##################
animal=Pancho
#question=CHAR_BASE_trial
#datelist=(230120 230125 230126 230127 220618 220627 220630 230119 230122)
question=PIG_BASE_stroke
which_level=stroke
#datelist=(230615 230620 230621 230622 230623 230626)
datelist=(230626)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 20s
done
#sleep 60m

