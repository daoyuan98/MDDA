#!/usr/bin/env bash

set -e

export PYTHONPATH=$PYTHONPATH:~/adda/adda

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
root_dataset=digit
sources=("synth")
targets=("mnist")

# model
model=digitbn
class_num=10
image_size=28

# settting
SOLVER=adam
iter=5000
batch_size=64
disc_lr=1e-5
encoder_lr=1e-7
display=10
snapshot=500
stepsize=1000001
adversary_layers=[3000,2000,1000]

source_exp_name=mnist
target_exp_name=mnist
classfier_exp_name=mnist
save_exp_name=mnist

head_path=/mdda
SAVE_PATH=$head_path/model/$root_dataset

set -e
cd $head_path/adda

for source in ${sources[*]}
do
    for target in ${targets[*]}
    do
    if [ $source != $target ]
    then
        echo $source "  --->   " $target
        #save path of encoder
        ENCODER_SAVE_PATH=${SAVE_PATH}/${source}/encoder/${source}_E_${target}_${model}_${save_exp_name}
        #save path of classifer
        CLASSIFER_SAVE_PATH=${SAVE_PATH}/${source}/domain/${source}_S_${target}_${model}_${save_exp_name}

        python3 $head_path/adda/tools/S2_adv.py \
                $source \
                $target \
                $model \
                $ENCODER_SAVE_PATH \
                $CLASSIFER_SAVE_PATH \
                $iter \
                $batch_size \
                $display \
                $disc_lr \
                $encoder_lr \
                $snapshot \
                $SOLVER \
                $class_num \
                $image_size \
                $root_dataset \
                $source_exp_name \
                $target_exp_name \
                $classfier_exp_name \
                $adversary_layers \
                $stepsize \
                $Lambda
    fi
    done
done



