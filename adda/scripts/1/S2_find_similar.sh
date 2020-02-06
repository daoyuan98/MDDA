set -e
export PYTHONPATH=$PYTHONPATH:~/adda/adda
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
head_path=/mdda
sources=("mnistm" "svhn" "synth")
root_dataset=digit

model=digitbn
encoder_exp_name=mnist
dis_exp_name=mnist
image_size=28

#save path
SAVE_PATH=$head_path/model/$root_dataset

echo "model: " $model
targets=('mnist')

for input_source in ${sources[*]}
do
    for target in ${targets[*]}
    do
    if [ $target != $input_source ]
    then
        source=$input_source
        dataset=$target

        #input path of source encoder
        SOURCE_ENCODER_INPUT_PATH=${SAVE_PATH}/${source}/encoder/${source}_E_${source}_${model}_${encoder_exp_name}
        #input path of target encoder
        TARGET_ENCODER_INPUT_PATH=${SAVE_PATH}/${source}/encoder/${source}_E_${target}_${model}_${encoder_exp_name}
        #input path of discriminator
        DISCRIMINATOR_INPUT_PATH=${SAVE_PATH}/${source}/domain/${source}_S_${dataset}_${model}_${dis_exp_name}
        #result images path /Result：
        RESULT_IMAGE_PATH=${SAVE_PATH}/${source}/result/${source}_R_${dataset}/find_similar

        python3 $head_path/adda/tools/find_similar_.py \
        ${source} $dataset $model $SOURCE_ENCODER_INPUT_PATH $TARGET_ENCODER_INPUT_PATH $DISCRIMINATOR_INPUT_PATH $RESULT_IMAGE_PATH $image_size $root_dataset

    fi
    done
done
