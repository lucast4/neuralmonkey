#!/bin/bash -e

###################
#animal=Diego
##question=CHAR_BASE_trial
##datelist=(231201 231204 231219 231130 231205 231207 231211 231220)
#question=PIG_BASE_stroke
#datelist=(230627 230629 230630)
##datelist=(230625 230626 230628)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#  sleep 20s
#done
#sleep 60m
#
###################
#animal=Pancho
##question=CHAR_BASE_trial
##datelist=(230120 230125 230126 230127 220618 220627 220630 230119 230122)
#question=PIG_BASE_stroke
#datelist=(230622 230623 230626)
##datelist=(230615 230620 230621)
#for date1 in "${datelist[@]}"
#do
#  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
#  touch ${logfile}
#  echo ${logfile}
#  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#  sleep 20s
#done
##sleep 60m


##################
animal=Diego
question=RULE_BASE_stroke
#datelist=(230723 230724 230726 230727 230728 230815 230816 230817 230913 230914 230730 231116 231118) # ALL
#datelist=(230724 230726 230727 230728 230815 230816 230817 230913 230914 230730) # subset that are good preprocesed.
datelist=(230724 230726 230817 230913 230730 231118) # Further subset, for rducing workload on copuater
#datelist=(230726 230913) # subset for running during day.
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 30m

##################
animal=Pancho
question=RULE_BASE_stroke
#datelist=(220831 220901 220902 220906 220907 220908 220909 230810 230811 230810 230829 230826 230824 231116 231114) # ALL
datelist=(220901 220908 220909 230810 230826) # Subset, to reduce workload,.
#datelist=(230810 230824) # subset for running during day.
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 1m
done
#sleep 60m


############################################### AnBm vs (AB)n
#################################################

##################
animal=Pancho
question=RULEVSCOL_BASE_stroke
datelist=(220829 220830)
#datelist=(230615 230620 230621)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_wrapper-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_wrapper.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 20s
done
#sleep 60m

