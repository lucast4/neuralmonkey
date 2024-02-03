#!/bin/bash -e

# SINGLE PRIMS
## PANCHO
datelist=(220715 220716 220717 220606 220608 220609 220610 220718 220719 220724 220918 221217 221218 221220 230103 230104) # the ones added 6/13/23, to get all days.
# animal="Pancho"

sleep 4h

## DIEGO
# datelist=(230603 230613 230614 230615 230616 230617 230618 230619 230621) # all
datelist=(230613 230621) # all
animal="Diego"
ANALY_VER="singleprim"
which_level="trial"

for date1 in "${datelist[@]}"
do
    logfile="../logs/log_analy_anova_script_loop_${date1}_${ANALY_VER}_${which_level}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${animal} ${date1} ${which_level} ${ANALY_VER} 2>&1 | tee ${logfile} &

	sleep 2m
done


