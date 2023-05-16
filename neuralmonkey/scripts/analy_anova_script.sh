#!/bin/bash

# params=( Pancho 220709 trial seqcontext )
# params=( Pancho 220709 trial seqcontext n)
# params=( Pancho 220709 trial seqcontext y)

# params=( Pancho 221020 trial rulesw )
# params=( Pancho 221020 trial rulesw n)
# params=( Pancho 221020 trial rulesw y)

# params=( Pancho 230105 trial seqcontext )
# params=( Pancho 230105 trial seqcontext n)
# params=( Pancho 230105 trial seqcontext y)

# params=( Pancho 220715 trial singleprim )
# params=( Pancho 220715 trial singleprim n)

params=( Pancho 220716 trial singleprim )
# params=( Pancho 220716 trial singleprim n)



# python analy_anova_extract.py ${params[@]}
# python analy_anova_plot.py ${params[@]}



# python analy_anova_extract.py ${params[@]}
python analy_anova_plot.py ${params[@]} n
python analy_anova_plot.py ${params[@]} y
