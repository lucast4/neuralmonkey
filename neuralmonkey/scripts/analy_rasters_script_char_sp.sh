#!/bin/bash -e


##################
animal=Pancho
datelist=(220628 220630 230126) 

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_char_sp-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_char_sp.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 30s
done
# sleep 180m
#  sleep 60m


###################
animal=Diego
datelist=(231205 231129 231201) # subset of final

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_rasters_script_char_sp-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_rasters_script_char_sp.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 30s
done

