# DL_project
Doubly convolutional neural networks. Reimplementation of the NIPS article : https://arxiv.org/abs/1610.09716

The version of python used is Python 2.7

# Datasets

- Dataset CIFAR10 (python version)
- Dataset CIFAR100 (python version)

https://www.cs.toronto.edu/~kriz/cifar.html

# Requirements

> pip install -r requirements.txt

# Setup environnement on tegner.pdc.kth.se


 ```bash
# move to DL_project on your local computer
# make sure you have the last version of the project
git pull

# get a karberos ticket for pdc
kinit -f uname_pdc@NADA.KTH.SE

# synchronise your local project to pdc
./pdc_script/rsync_DL_project.sh uname_pdc

# login on pdc
ssh uname_pdc@tegner.pdc.kth.se

# move to DL_project
cd DL_project

# create and setup virtual env
./pdc_script/install_virtualenv.sh
 ```

# Run a job on pdc

 ```bash
# login on pdc
ssh uname_pdc@tegner.pdc.kth.se

# move to DL_project
cd DL_project

# you may want to change some configuration in ./pdc_script/submitjob.sh
# [BERTRAND] set the course code name to DT2424
# you can choose the gpu between K420 or K80

# create and setup virtual env
# dataset in [cifar10, cifar100, cifar10_augmented, cifar100_augmented]
./pdc_script/submitjob.sh dataset

# check the job status
squeue -u uname_pdc
```

# Extra

```bash
# kill a job if needed
# jobID can be get with the previous command
scancel jobID
```

