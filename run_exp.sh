#!/bin/bash

source ~/pyt/bin/activate

python ~/bio-challenge/src/run.py --device cuda \
					--save_dir ~/bio-challenge/.model/ --save_preds --save_model \
					--model trial_7 \
					--epoch 50 --bidir


# For evaluating prediction by competition script
#python ~/bio-challenge/task3/src/evaluation.py ref/ res/