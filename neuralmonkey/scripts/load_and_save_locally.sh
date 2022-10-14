#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt
# ./load_and_save_locally.sh 2>&1 | tee preprocess_log.txt


# Mount server
server_mount

touch preprocess_log.txt

####### Version 1- give a list of dates
# datelist=( 220702 220703 220630 220628 220624 220616 220603 220609 )
# datelist=( 220704 )
# datelist=( 220606 220608 220609 220610 220715 220716 220717 )
# for date in "${datelist[@]}"
# do
# 	# echo ${date}
# 	# python -m neuralmonkey.scripts.load_and_save_locally ${date} 2>&1 | tee load_and_save_locally_log_${date}.txt & 
# 	python -m neuralmonkey.scripts.load_and_save_locally ${date}
# done



# ######## version 1 - iterate thru dates in order.
# d=220526 # first date to check.
# date2=220815 # the final date to check

# d=220603 # first date to check.
# d=220707 # the final date to check

d=220815 # first date to check.
date2=221004 # the final date to check
date1=$d
until [[ ${date1} > ${date2} ]]; do
  echo ${date1}
  # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
  python -m neuralmonkey.scripts.load_and_save_locally ${date1} |& tee -a preprocess_log.txt
  date1=$(date +%y%m%d -d "${date1} + 1 day")
done
