#!/bin/bash -e


# ##################
animal=Pancho
datelist=(230810 230811 230824 230826 230829 231114 231116 240830 220831 220901 250321 250322 220902 220906 220907 220908 220909) # ALL (confirmed)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_trial-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_trial.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 30s
done
sleep 10m

##################
animal=Diego
datelist=(230728 231118 240822 230723 230724 230726 230727 230730 230815 230816 230817 230913 230914 230915 231116 240827 250319 250321) # ALL (confirmed)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_trial-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_trial.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 30s
done

sleep 20m

