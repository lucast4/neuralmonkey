#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt

#######################################
# datelist=( 220702 220703 220630 220628 220624 220616 220603 220609 )
# datelist=( 220704 )
# datelist=( 220702 220705 220706 220707 220708 220709 220603 220608 220609 220616 220624 )
# for date in "${datelist[@]}"
# do
# 	# echo ${date}
# 	# python -m neuralmonkey.scripts.load_and_save_locally ${date} 2>&1 | tee load_and_save_locally_log_${date}.txt & 
# 	python -m neuralmonkey.scripts.load_and_save_locally ${date}
# done

# date=220703
# python -m neuralmonkey.scripts.load_and_save_locally ${date}

# d=220603 # first date to check.
# date2=220707 # the final date to check
d=220526 # first date to check.
# date2=220720 # the final date to check
# d=220720 # first date to check.
date2=220828 # the final date to check
date1=$d
until [[ ${date1} > ${date2} ]]; do
  echo ${date1}
  python -m neuralmonkey.scripts.load_and_save_locally ${date1}
  date1=$(date +%y%m%d -d "${date1} + 1 day")
done
