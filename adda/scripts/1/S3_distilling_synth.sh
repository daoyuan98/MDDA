#==Dataset==
dataset=digit
source_domain=("synth")
target_domain=("mnist")
class_num=10
image_size=28
adda_encoder=1
ft_classifer=0
          
#==model===
model=digitbn
encoder_exp_name=mnist
classifer_exp_name=mnist
exp_name=mnist
head_path="/mdda"

#==setting==
lr=0.00001
stepsize=80
snapshot=10
solver=sgd
iterations=30
batch_size=64
display=2
#==Dont Change==
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

for source in ${source_domain[*]}
do
    for target in ${target_domain[*]}
    do
    if [ $source != $target ]
    then
        result_image_path=$head_path/model/digit/${source}/result/${source}_R_${target}/find_similar
        echo $result_image_path
        echo S $source T $target
        python3 $head_path/adda/tools/S3_distilling.py ${source} \
        ${target} \
        ${model} \
        ${dataset} \
        ${class_num} \
        ${encoder_exp_name} \
        ${classifer_exp_name} \
        ${exp_name} \
        ${result_image_path} \
        $adda_encoder \
        $ft_classifer \
        $lr \
        $stepsize \
        $snapshot \
        $solver \
        $iterations \
        $batch_size \
        $display
    fi
    done
done
