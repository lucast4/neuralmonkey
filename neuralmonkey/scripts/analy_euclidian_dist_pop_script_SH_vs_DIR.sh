#!/bin/bash -e

##################
animal=Diego
question=RULESW_ANBMCK_DIR_STROKE
which_level=stroke
#datelist=(230917 230823 230804 230809 230825 230813 230827 230919) # AnBmCk vs. DIR
#datelist=(230705 230703 230711 230713 230719) # ABC vs. DIR
datelist=(230719 230823 230804 230827 230919) # quick subset good
#datelist=(230823) # testing, 4/9
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 50m

animal=Pancho
question=RULESW_ANBMCK_DIR_STROKE
which_level=stroke
#datelist=(221023 230910 230912 230914 230919)
datelist=(221023 230910 230914 230919) # quick subset good
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_dist_pop-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_dist_pop_script.py ${animal} ${date1} ${question} ${which_level} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 40m
