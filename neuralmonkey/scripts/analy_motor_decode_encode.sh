#!/bin/bash -e


###################
animal=Diego
datelist=(240508 230614 230615 230618 230619) # [A and B] [ignoring 231121]
question=SP_BASE_stroke
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_motor_decode_encode-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_motor_decode_encode.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 10s
done
# sleep 180m
sleep 10m

###################
animal=Pancho
datelist=(220715 220724 220716 220717 240530) # [A and B] [ignoring 231121]
question=SP_BASE_stroke
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_motor_decode_encode-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_motor_decode_encode.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 10s
done
# sleep 180m
sleep 10m


# ###################
# animal=Diego
# datelist=(231220 231205 231122 231128 231129 231201 231120 231206 231218) # [A and B] [ignoring 231121]
# question=CHAR_BASE_stroke
# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_motor_decode_encode-${animal}_${date1}_${question}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_motor_decode_encode.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#   sleep 10s
# done
# # sleep 180m
# sleep 10m

# ###################
# animal=Pancho
# datelist=(220614 220616 220621 220622 220624 220627 220618 220626 220628 220630) # [A and B] [ignoring 231121]
# question=CHAR_BASE_stroke
# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_motor_decode_encode-${animal}_${date1}_${question}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_motor_decode_encode.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#   sleep 10s
# done
# # sleep 180m
# sleep 10m

