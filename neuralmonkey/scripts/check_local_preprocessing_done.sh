#!/bin/bash

shopt -s expand_aliases

BASEDIR="/gorilla3/neural_preprocess/recordings/Pancho"
animal="Pancho"
alias echo="echo -e"

# ascii codes - for success/error message colorcoding
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

ERROR="${RED}ERROR${NC}"
SUCCESS="${GREEN}SUCCESS${NC}"

cd ${BASEDIR}
folders=$(find . -maxdepth 2 -type d -name "${animal}-*" | sort -nr) 
echo ${folders}

for f in ${folders}
do
    echo "" #newline
    echo " ============================== "
    echo "Checking neural data folder: ${f}"
    cd $f
    [ -z $(find . -type f -name "data_datall.pkl") ] && echo "     |--${ERROR}: data_datall.pkl not found" || echo "     |--${SUCCESS}: data_datall.pkl found"
    [ -z $(find . -type f -name "data_spikes.pkl") ] && echo "     |--${ERROR}: data_spikes.pkl not found" || echo "     |--${SUCCESS}: data_spikes.pkl found"
    [ -z $(find . -type f -name "data_tank.pkl") ] && echo "     |--${ERROR}: data_tank.pkl not found" || echo "     |--${SUCCESS}: data_tank.pkl found"
    [ -z $(find . -type f -name "mapper_st2dat.pkl") ] && echo "     |--${ERROR}: mapper_st2dat.pkl not found" || echo "     |--${SUCCESS}: mapper_st2dat.pkl found"
    ls -lhS . | awk '{print $5, $9}'
    
    spikes_tdt_quick=$(find . -maxdepth 1 -type d -name "spikes_tdt_quick*")
    if [[ -z $spikes_tdt_quick ]]; then
        echo "--${ERROR}: no spikes_tdt_quick folder found in ${f}"
        cd ../..
    else
        echo "--${SUCCESS}: found spikes_tdt_quick folder found in ${f}"
        # echo $(ls)
        # asdfsdaf
        cd $spikes_tdt_quick
        spike_png_files=($(find . -type f -name "*RSn2-*.png"))
        num_files=${#spike_png_files[@]}
        if [[ ! num_files -eq 256 ]]; then
          echo "--${ERROR}: found ${num_files} RSn2 files, expecting 256"
        else
          echo "--${SUCCESS}: found ${num_files} RSn2 files, expecting 256"
        fi

        spike_png_files=($(find . -type f -name "*RSn3-*.png"))
        num_files=${#spike_png_files[@]}
        if [[ ! num_files -eq 256 ]]; then
          echo "--${ERROR}: found ${num_files} RSn3 files, expecting 256"
        else
          echo "--${SUCCESS}: found ${num_files} RSn3 files, expecting 256"
        fi

        cd ../../..
    fi

    # echo "CURRENT PATH: $(pwd)"

done

