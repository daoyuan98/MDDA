# Multi-Source Distilling Domain Adaptation

Offcial implementation of AAAI2020 paper, Multi-source distilling domain adaptation[https://arxiv.org/abs/1911.11554]

![image](https://github.com/daoyuan98/MDDA/images/MDDA.png)

## Libraries

python 3.6.10

tensorflow 1.12.0

Besides, other libraries are needed for better display

* tqdm
* colorlog
* yaml

## Data Preprocessing

Since the sources in digits5 experiments are in different formats, we preprocess all of these data into a **json** file and each image is encoded into an base64 string with the library ``cv2`` and ``base64``. We preprocess data in office-31 experiments in the same way.

## Execution

As illustrated in the above figure,  There are mainly 4 stages in our framework. In this repo, we give an experiments on ``usps`` as an example. To execute them, you may execute the following steps sequentially. All of these scripts are in the directory of ``scripts``.

#### 1. Source classifier pre-training

Pretrain source encoders and classifiers in each source, respectively.

```
./S1_train_mnistm.sh
./S1_train_svhn.sh
./S1_train_synth.sh
./S1_train_usps.sh
```

#### 2. Adversarial discriminative adaptation

Using adversarial training to train an target encoder that can align source and target features with the supervision of a discriminator.

```
./S2_adv_mnistm.sh
./S2_adv_svhn.sh
./S2_adv_synth.sh
./S2_adv_usps.sh
```

After adversarial training, the next step is stage3 in distilling. But before stage 3, we should find source samples that used for distilling. Therefore, we should execute the following script:

```
./S2_find_similar.sh
```

This script will find similar samples from each source and save them as ``.npy`` file. Also, this step will save discriminator outputs that will be used in stage 4 for results aggregation.

#### 3. Source distilling

This step, in each domain, we will load source encoder and source classifier pretrained in stage 1. Then, using data found in stage 2, we will fix the weights of encoder and fine-tune the source classifier.

```
./S3_distilling_mnistm.sh
./S3_distilling_svhn.sh
./S3_distilling_synth.sh
./S3_distilling_usps.sh
```

#### 4. Aggregate target prediction

In this step, we will get the output of models(adversarial trained encoder and distilled classifier) in each source given target data.

```
./S4_eval.sh
```

Then, we will get the output of each source models. Then, we change directory to adda/tools/ and run ``voting.py` with python, we will get the final result.



## Acknowledgement

 This repo has referenced some codes from ADDA[https://github.com/erictzeng/adda]



If you have any questions about this implementation, feel free to contact me at gzwang98@gmail.com