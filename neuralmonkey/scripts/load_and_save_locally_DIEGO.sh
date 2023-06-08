#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt

# cd /gorilla1/code/neuralmonkey/neuralmonkey/scripts
# ./load_and_save_locally.sh 2>&1 | tee preprocess_log_221108.txt


# Mount server
server_mount

touch preprocess_log.txt
animal=Diego
# ####### Version 1- give a list of dates
#230524
datelist=( 230603 )

for date1 in "${datelist[@]}"
do
    echo ${date1}
    # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
    python -m neuralmonkey.scripts.load_and_save_locally ${date1} ${animal} |& tee -a preprocess_log.txt
done


# ######## version 1 - iterate thru dates in order.
# # 230525
# d=220526 # first date to check.
# date2=230403 # the final date to check

# date1=$d
# until [[ ${date1} > ${date2} ]]; do
#   echo ${date1}
#   # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
#   python -m neuralmonkey.scripts.load_and_save_locally ${date1} |& tee -a preprocess_log.txt
#   date1=$(date +%y%m%d -d "${date1} + 1 day")
# done