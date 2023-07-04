#!/bin/bash -e

# PRIMS IN GRID (SEQUENCE)
# datelist=(220709 220714 220805 230105 230106 230108 230109) # all
# datelist=( 230105 )
# datelist=(220709 220714 220727 220731 220805 230106)
# datelist=(220709 220714 230106)

# sleep 2h
# datelist=(230105 230106 230108 230109) # 6/6/23 - the well-structured experiments
datelist=(230105 230106 230108 230109 230615 230616 230620 230621 230622 230623 230626) # 6/28/23 - the well-structured experiments, including the new expts

ANALY_VER="seqcontext"
which_level="trial"
animal=Pancho

for date1 in "${datelist[@]}"
do
    logfile="../logs/log_analy_anova_script_loop_${date1}_${ANALY_VER}_${which_level}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${animal} ${date1} ${which_level} ${ANALY_VER} 2>&1 | tee ${logfile} &

	sleep 1m
done


