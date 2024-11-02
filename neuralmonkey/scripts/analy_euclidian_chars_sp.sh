#!/bin/bash -e


animal=Pancho
# datelist=(230122 230125 230126 230127 230112 230117 230118 230119 230120) # ALL

# datelist=(230119 230120 230122 230125 230126 230127) # [1] Those with decent SP
# datelist=(220531 220602 220603 220618 220626 220628 220630) # [2] Getting more dates, so that combined with previous, is all.

datelist=(230119 230120 230122 230125 230126 230127 220531 220602 220603 220618 220626 220628 220630) # All [1,2]

# combine=1 # GOOD? 
combine=0 # Just for heatmaps, managable sized plots.
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_chars_sp-${animal}_${date1}_${combine}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_chars_sp.py ${animal} ${date1} ${combine} 2>&1 | tee ${logfile} &
  sleep 30s
done
# sleep 180m
#  sleep 60m


###################
animal=Diego
# datelist=(230724 230726 230730 230816 230817 230913 230914 230915 231116 231118) # ALL
# datelist=(231211) # ONE
datelist=(231130 231205 231211 231122 231128 231129 231201 231213 231204) # MORE
combine=1
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_chars_sp-${animal}_${date1}_${combine}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_chars_sp.py ${animal} ${date1} ${combine} 2>&1 | tee ${logfile} &
  sleep 30s
done

