#!/bin/bash -e

# SINGLE PRIMS
## PANCHO
# datelist=(220606 220608 220610 220715 220716 220717 220918)
# # datelist=( 220715 220716 220717 )
# # datelist=(220715 220716)
# ANALY_VER="singleprim"
# which_level="trial"
# animal="Pancho"

## DIEGO
datelist=(230603)
ANALY_VER="singleprim"
which_level="trial"
animal="Diego"

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}_${which_level}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${animal} ${date1} ${which_level} ${ANALY_VER} 2>&1 | tee ${logfile} &
done

