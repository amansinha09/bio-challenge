#!/bin/bash

source ~/pyt/bin/activate

python ~/bio-challenge/src/run.py --device cuda \
					--save_dir ~/bio-challenge/.model/ --save_preds --save_model \
					--model baseline_rnn_12 \
					--epoch 20


# For evaluating prediction by competition script
#python ~/bio-challenge/task3/src/evaluation.py ref/ res/
