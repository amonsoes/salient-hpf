# HPFAttack

This repository holds the code for the paper "Combining Frequency-Based Smoothing and Salient Masking for Performant and Imperceptible Adversarial Samples". It contains commands to replicate all experiments and the survey results.

**Disclaimer:** This repository was not optimized for speed but to provide a proof of concept. To have the most realistic adversarial attack
transform possible, the attack will be applied after scaling and before model-specific transformations. That means the attack cannot be processed
in batches, which is much faster and how the Torchattacks library usually applies adversarial attacks. A more user-friendly and performant version will follow shortly. 

<br />

## Supported Datasets

- Nips2017 Adversarial Challenge ImageNet Subset. [Find Here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/overview)

<br />

## Models

- ResNet: Pretrained IMGNet model. [He et al 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- Inception: Pretrained IMGNet model with depthwise-separable convolutions. [Szegedy et al 2015](https://arxiv.org/pdf/1512.00567.pdf)


<br />

## Requirements

(1) **Install module requirements**

All experiments performed on Python 3.10, torch 2.0+cu117 and torchvision 0.15.1+cu117
Download and install torch + torchvision [here](https://pytorch.org/)

Install remaining modules:

```
pip install -r requirements.txt
```

(2) **Download Datasets**


- Nips2017 Adversarial Challenge ImageNet Subset. [Find Here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/overview)

Download the datasets for the specified sources.

(3 - 6) **Set-up by helper file**

You can set up the repository with the required data, models and package extensions by calling the helper file.
Additionally you'll have to provide the paths to the two zipped datasets (starting with 140k_flickr_faces)

```
python3 setup_tool.py --nips17_dataset_path=path/to/nips17.zip --extend_modules=True
```

make sure you have sudo rights and 'unzip' installed (sudo apt-get install unzip).
If that does not work you can alternatively do steps (3) - (6) manually with the following:

(3) **Data set-up**

Download both dataset and place them in a folder called 'data' in the root of the repository.


(4) **Pretrained model set-up**

Pretrained models can be downloaded [here](https://drive.google.com/file/d/1tewpsKAbpud6RTwGTT8_fDa5EFYTdbRW/view?usp=share_link).
Put downloaded folder in a folder './saves' and provide the path to the model.pt-file for the respective scripts.


(5) **Extend torchattacks and other modules**

torchattacks is a repository to conduct adversarial attacks in a PyTorch environment. The base repository has been
extended.

- Replace attack folder in package_extensions with the one in your torchattacks package.
- Replace _dct.py in package_extensions with the one in your torch_dct package.
- Replace MAD.py in package_extensions with the one in your IQA_pytorch package.



(6) **Download and install pytorch-colors**

Get repository [here](https://github.com/jorge-pessoa/pytorch-colors) and follow the installation guideline.

<br />

## Replicate Experiments

In this paper, you can extend 5 Attacks, but not every extension is available for every attack:

FGSM:

- HPF
- LF-Boosting
- CbCr-boosting

BIM:

- HPF
- LF-Boosting

VMIFGSM:

- HPF
- LF-Boosting

PGDL2:

- HPF
- LF-Boosting

**Important**: If you test a hpf attack on a cifar dataset, set the option "--dct_patch_size=4" to adjust for the smaller images.


### (1) White-Box Attacks

Choose an attack and appropriate epsilon to replicate an attack from the experiments.

ATTACK:

- fgsm
- hpf_fgsm
- bim
- hpf_bim
- vmifgsm
- hpf_vmifgsm

EPSILON:

- fgsm : 0.129
- bim : 0.023
- vmifgsm: 0.008

DATASET:

- nips17
- cifar100

```bash
python3 run_pretrained.py --dataset=DATASET --model_name=resnet --transform=pretrained --device=cuda:0 --batchsize 32 --adversarial=True --spatial_adv_type=ATTACK --eps=EPSILON --use_sal_mask Tru
```

<br>

### (2) Black-Box Attacks

ATTACKS:

- pg_rgf
- hpf_pg_rgf
- vmifgsm
- hpf_vmifgsm

add 'hpf_' to attack with hpf version

**PG-RGF**

```bash
python3 run_pretrained.py --dataset DATASET --model_name inception --transform pretrained --adversarial True --spatial_adv_type pg_rgf --eps 7.0 --use_sal_mask True --surrogate_model resnet --max_queries 10
```

**VMIFGSM**

```bash
python3 run_pretrained.py --dataset DATASET --model_name inception --transform pretrained --adversarial True --spatial_adv_type vmifgsm --eps 0.07 --use_sal_mask True --surrogate_model resnet
```

<br>

### (3) Adv Training Experiments 

Adversarial Training Protocols Available:

- PGD-Resnet
- FBF-Resnet

FGSM : eps 0.0129
BIM : eps 0.01
VMIFGSM: eps 0.008

DATASET:

- nips17
- cifar10 (do not forget to change the path in --pretrained to a cifar10 model)

change attack and eps value accordingly to get paper results

**PGD-Resnet**

```bash
python3 run_pretrained.py --dataset cifar10 --transform pretrained --adversarial_pretrained True --adv_pretrained_protocol pgd --batchsize 16 --device cuda:0 --model_name adv_resnet_pgd --adversarial True --spatial_adv_type fgsm --eps 0.036 --surrogate_model adv_resnet_pgd --use_sal_mask True
```

**FBF-Resnet**

```bash
python3 run_pretrained.py --dataset DATASET --pretrained=./saves/models/Adversarial/fbf_models/imagenet_model_weights_2px.pth.tar --transform pretrained --adversarial_pretrained True --adv_pretrained_protocol fbf --batchsize 16 --device cuda:0 --model_name adv-resnet-fbf --adversarial True --spatial_adv_type hpf_fgsm --eps 0.0129 --surrogate_model adv_resnet_fbf --use_sal_mask True
```

<br>

### (4) NIPS 2017 Adversarial Challenge Benchmark

**Inception-V3**

Without HPF extension:

```bash
python3 run_pretrained.py --dataset=nips17 --model_name=inception --transform=pretrained--device=cuda:0 --batchsize 32 --adversarial=True --spatial_adv_type=pgdl2 --eps=5.0 --alpha=3.0 --surrogate_model inception
```

With HPF extension:

```bash
python3 run_pretrained.py --dataset=nips17 --model_name=inception --transform=pretrained --device=cuda:0 --batchsize 32 --adversarial=True --spatial_adv_type=hpf_pgdl2 --eps=5.0 --alpha=3.0 --use_sal_mask True --surrogate_model inception
```

**Adversarially augmented Inception-V3**

Without HPF extension:

```bash
python3 run_pretrained.py --dataset=nips17 --model_name=adv-inception --transform=pretrained --device=cuda:1 --batchsize 32 --adversarial=True --spatial_adv_type=pgdl2 --eps=5.0 --alpha=3.0 --surrogate_model adv-inception
```

Wit HPF extension:

```bash
python3 run_pretrained.py --dataset=nips17 --model_name=adv-inception --transform=pretrained --device=cuda:1 --batchsize 32 --adversarial=True --spatial_adv_type=hpf_pgdl2 --eps=5.0 --alpha=3.0 --use_sal_mask True --surrogate_model adv-inception
```

<br>

### (5) Compression Experiments

replace eps argument with 0.04 for experiments with higher perturbation

1. **FGSM**

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type fgsm --eps=0.0129 --attack_compression True --attack_compression_rate 50 --use_sal_mask True --surrogate_model resnet --batchsize 32
```

2. **HPF-FGSM**

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type hpf_fgsm --eps=0.0129 --attack_compression True --attack_compression_rate 50 --use_sal_mask True --surrogate_model resnet --batchsize 32
```

3. **HPF-FGSM (LF-boosted)**

```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type hpf_fgsm --eps=0.0129 --attack_compression True --attack_compression_rate 50 --use_sal_mask True --surrogate_model resnet --batchsize 32 --lf_boosting 0.5
```



### (6-7) MAD Experiments / Survey Data

MAD will be reported in the respective log-folder.

Get results of survey by processing the CSV of the survey:

```bash
python3 ./survey/process_survey_data.py
```