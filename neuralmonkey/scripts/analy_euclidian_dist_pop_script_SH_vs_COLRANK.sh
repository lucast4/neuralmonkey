#!/bin/bash -e

##################
animal=Diego
question=RULESW_ANBMCK_COLRANK_STROKE
which_level=stroke
#datelist=(230910 230911 230907 230912 230927 231001) # ALL
#datelist=(230910 230907 230912 231001) # subset, just to do during day
#datelist=(230910 230912) # quick, good
#datelist=(230912) # quick, good (subset)
#datelist=(230911) # testing, 4/9
datelist=(230910 230912 230927 231001) # GOOD SUBSET..

# To rerun entirely
# datelist=(231001)
# # To plot
# datelist=()

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 1m
done
# sleep 10m
sleep 90m

 animal=Pancho
 question=RULESW_ANBMCK_COLRANK_STROKE
 which_level=stroke
 datelist=(230928 230929)

 # To rerun entirely
 datelist=()
 # # To plot
 # datelist=()

 for date1 in "${datelist[@]}"
 do
   logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
   touch ${logfile}
   echo ${logfile}
   python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
   sleep 1m
 done
 sleep 30m
