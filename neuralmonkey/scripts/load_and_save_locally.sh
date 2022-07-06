# Run this in base dir: /neuralmonkey

#!/bin/bash

# datelist=( 220702 220703 220630 220628 220624 220616 220603 220609 )
datelist=( 220704 )
for date in "${datelist[@]}"
do
	# echo ${date}
	# python -m neuralmonkey.scripts.load_and_save_locally ${date} 2>&1 | tee load_and_save_locally_log_${date}.txt & 
	python -m neuralmonkey.scripts.load_and_save_locally ${date}
done

# date=220703
# python -m neuralmonkey.scripts.load_and_save_locally ${date}

