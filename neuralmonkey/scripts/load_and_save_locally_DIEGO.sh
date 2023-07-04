#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt

# cd /gorilla1/code/neuralmonkey/neuralmonkey/scripts
# ./load_and_save_locally_DIEGO.sh 2>&1 | tee preprocess_log_DIEGO_221108.txt


# Mount server
# server_mount
freiwald_mount

animal=Diego
# touch preprocess_log.txt

# ####### Version 1- give a list of dates
# #230524
# datelist=( 230603 )

# for date1 in "${datelist[@]}"
# do
#     echo ${date1}
#     # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
#     python -m neuralmonkey.scripts.load_and_save_locally ${date1} ${animal} |& tee -a preprocess_log.txt
# done


######## version 1 - iterate thru dates in order.
# 230617
# d=230603 # first date to check.
# date2=230616 # the final date to check

# # 230621
# d=230614 # first date to check.
# date2=230616 # the final date to check

# 230625_2
# d=230617 # first date to check.
# date2=230621 # the final date to check

# 230625_3
d=230521 # first date to check.
date2=230613 # the final date to check

date1=$d
until [[ ${date1} > ${date2} ]]; do
  echo ${date1}
  python -m neuralmonkey.scripts.load_and_save_locally ${date1} ${animal}
  date1=$(date +%y%m%d -d "${date1} + 1 day")
done