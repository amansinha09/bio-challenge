#!/bin/bash

source ~/pyt/bin/activate

python ~/bio-challenge/task3/src/run.py --device cuda --save ~/bio-challenge/task3/.model/  --model trial_4 --epoch 10
