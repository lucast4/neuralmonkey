#!/bin/bash -e


# ##################
# animal=Diego
# # datelist=(230614 230615) # good ones
# # datelist=(230615) # good ones

# datelist=(230614 230615 240508 240509 240510 240513 240530) # The remaining ones (not good)
# # datelist=(240513) # Failures
# var_other=seqc_0_loc

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_shape_invariance_all_plots_SP-${animal}_${date1}_${var_other}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_shape_invariance_all_plots_SP.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
#   sleep 30s
# done

# animal=Diego
# datelist=(230618 230619 240510 240530) # This completes it
# # datelist=(230618 230619) # good ones
# # datelist=(230618) # good ones
# var_other=gridsize

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_shape_invariance_all_plots_SP-${animal}_${date1}_${var_other}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_shape_invariance_all_plots_SP.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
#   sleep 30s
# done
# sleep 1h

##################
animal=Pancho
# datelist=() # Further subset, for rducing workload on copuater
# datelist=(220608 220715 220717) # good ones
# datelist=(220715) # good ones

# datelist=(220606 220608 220715 220717 220724 220918 221218 240508 240509 240510 240515 240530) # The remaining
datelist=(220715 220724) # Just to check PMvl

var_other=seqc_0_loc

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done

animal=Pancho
# datelist=(220716 220717) # good ones [DONE]
# datelist=(220716) # good ones [DONE]
datelist=(220716 220717 240530) # good ones [DONE]

var_other=gridsize

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_shape_invariance_all_plots_SP-${animal}_${date1}_${var_other}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_shape_invariance_all_plots_SP.py ${animal} ${date1} ${var_other} 2>&1 | tee ${logfile} &
  sleep 30s
done