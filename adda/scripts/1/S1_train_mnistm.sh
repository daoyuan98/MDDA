#!/usr/bin/env bash

root_dataset=digit
sources=("mnistm")
target=mnist
class_num=10

#=MODEL==
model=digitbn
exp_name=mnist

#==Argument===
SOLVER=sgd
stepsize=5000
train_adda=0
iter=3000
bs=64
display=50
lr=0.01
snapshot=500
display=50
stepsize=1000

set -e
set -x
export PYTHONPATH=$PYTHONPATH:~/adda/adda
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /mdda/adda/tools/
for source in ${sources[*]}
do
    python3 S1_train.py $source $target train $model $root_dataset $class_num $exp_name \
           --iterations $iter \
           --batch_size $bs \
           --display $display \
           --lr $lr \
           --snapshot $snapshot \
           --stepsize $stepsize \
           --solver $SOLVER
done
