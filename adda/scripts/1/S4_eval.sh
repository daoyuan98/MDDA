#==Setting==
root_dataset=digit
source_domain=("usps" "mnistm" "svhn" "synth")
target_domain=("mnist")
class_num=10
train_adda=0

model=digitbn
exp_name=mnist
image_size=28
adda_encoder=1
ft_classifer=0
          
#==Dont Change==
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /mdda/adda/tools
for source in ${source_domain[*]}
do
    for target in ${target_domain[*]}
    do
        echo S $source T $target
        python eval.py ${target} test ${model} ${source} ${root_dataset} ${exp_name} ${image_size} ${class_num}\
        --adda_encoder $adda_encoder \
        --ft_classifer $ft_classifer \
        --train_adda $train_adda
    done
done
