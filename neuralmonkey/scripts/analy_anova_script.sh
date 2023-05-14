#!/bin/bash

# params=( Pancho 220709 trial seqcontext n)
# params=( Pancho 220709 trial seqcontext y)
# params=( Pancho 221020 trial rulesw n)
# params=( Pancho 221020 trial rulesw y)
# params=( Pancho 230105 trial seqcontext n)
params=( Pancho 230105 trial seqcontext y)

# python analy_anova_extract.py ${params[@]}
python analy_anova_plot.py ${params[@]}
