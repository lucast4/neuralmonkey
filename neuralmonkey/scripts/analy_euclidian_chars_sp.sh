#!/bin/bash -e

###################
animal=Diego
# datelist=(230724 230726 230730 230816 230817 230913 230914 230915 231116 231118) # ALL
# datelist=(231211) # ONE
# datelist=(231130 231211 231213 231204) # [C] original, has only one SP bloque
# datelist=(231205 231122 231128 231129 231201) # [A] FINAL (has sp across mult bloques)
# datelist=(231120 231121 231206 231218 231220) # [B] ADDED (leaving out 231207, lacking preprocess) [only one SP bloque]
# datelist=(231205 231122 231128 231129 231201 231120 231121 231206 231218 231220) # [A and B]
datelist=(231205 231122 231128 231129 231201 231120 231206 231218 231220) # [A and B] [ignoring 231121]
# datelist=(231122 231206) # 

combine=1
trial_ver=0
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_chars_sp-${animal}_${date1}_${combine}_${trial_ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_chars_sp.py ${animal} ${date1} ${combine} ${trial_ver} 2>&1 | tee ${logfile} &
  sleep 10s
done
# sleep 180m
sleep 30s

###################
animal=Pancho
# datelist=(230122 230125 230126 230127 230112 230117 230118 230119 230120) # ALL

# --- Not combined (has all the days)
# datelist=(230119 230120 230122 230125 230126 230127) # [1] Those with decent SP
# datelist=(220531 220602 220603 220618 220626 220628 220630) # [2] Getting more dates, so that combined with previous, is all.
# datelist=(230119 230120 230122 230125 230126 230127 220531 220602 220603 220618 220626 220628 220630) # All [1,2]
# combine=0 # Just for heatmaps, managable sized plots.

# --- Combined areas (only subset of days)
# datelist=(220618 220626 220628 220630 230119 230120 230126 230127) # [A] All in set 1
# datelist=(220614 220616 220621 220622 220624 220627 230112 230117 230118) # [B] Extra to set 1 (This COMPLETES all days)
# datelist=(220618 220626 220628 220630 230119 230120 230126 230127 220614 220616 220621 220622 220624 220627 230112 230117 230118) # [A+B] Combined

datelist=(220614 220616 220621 220622 220624 220627 220618 220626 220628 220630) # Just final, to get PMVl
# datelist=(220614) # Failed

combine=1
trial_ver=0

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_euclidian_chars_sp-${animal}_${date1}_${combine}_${trial_ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_euclidian_chars_sp.py ${animal} ${date1} ${combine} ${trial_ver} 2>&1 | tee ${logfile} &
  sleep 10s
done

