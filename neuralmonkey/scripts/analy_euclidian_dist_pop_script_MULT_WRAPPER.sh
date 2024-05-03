#!/bin/bash -e

# python analy_euclidian_dist_pop_script_MULT.py Diego AnBmCk_general &
# python analy_euclidian_dist_pop_script_MULT.py Diego sh_vs_seqsup &
# python analy_euclidian_dist_pop_script_MULT.py Diego sh_vs_dir &
python analy_euclidian_dist_pop_script_MULT.py Diego sh_vs_col &
python analy_euclidian_dist_pop_script_MULT.py Pancho two_shape_sets &

sleep 10m;

python analy_euclidian_dist_pop_script_MULT.py Pancho AnBmCk_general
python analy_euclidian_dist_pop_script_MULT.py Pancho sh_vs_seqsup
# python analy_euclidian_dist_pop_script_MULT.py Pancho sh_vs_dir
python analy_euclidian_dist_pop_script_MULT.py Pancho sh_vs_col
