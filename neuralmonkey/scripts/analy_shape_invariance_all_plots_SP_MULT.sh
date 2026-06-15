#!/bin/bash -e


##################
animal=Diego
datelist=(230614 230615) # good ones
var_other=seqc_0_loc

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP_MULT-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP_MULT.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done

animal=Diego
datelist=(230618 230619) # good ones
var_other=gridsize

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP_MULT-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP_MULT.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done

##################
animal=Pancho
datelist=(220608 220715 220717) # good ones
var_other=seqc_0_loc

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP_MULT-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP_MULT.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done

animal=Pancho
datelist=(220716 220717) # good ones [DONE]
var_other=gridsize

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP_MULT-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP_MULT.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done
