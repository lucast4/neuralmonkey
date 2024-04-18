#!/bin/bash -e

###################
animal=Diego
question=RULESW_ANY_SEQSUP_STROKE
which_level=stroke
#datelist=(230922 230920 230921) # SH vs DIR vs SEQSUP
#datelist=(230924 230925) # SH vs DIR vs SEQSUP vs COLOR_RANK
#datelist=(230922 230924) # quick, test of all
 datelist=(230922 230920 230921 230924 230925) # ALL

# To rerun entirely
#datelist=(230922 230920 230924)
# # To plot
# datelist=()

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 2m
done
sleep 30m
#

 animal=Pancho
 question=RULESW_ANY_SEQSUP_STROKE
 which_level=stroke
 #datelist=(230923 230921 230920) # SH vs DIR vs SEQSUP
 #datelist=(231019 231020) # SH vs DIR vs SEQSUP vs COLOR_RANK
 datelist=(230923 230921 230920 231019) # ALL
 #datelist=() # quick, test of all

 # To rerun entirely
# datelist=(230923 230921 231019)
 # To plot
 datelist=()

 for date1 in "${datelist[@]}"
 do
   logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
   touch ${logfile}
   echo ${logfile}
   python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
   sleep 1m
 done
 sleep 60m
