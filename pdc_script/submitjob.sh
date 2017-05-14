#!/bin/bash
# usage : submitjob.sh database_name
# database_name in [cifar10, cifar100, cifar10_augmented, cifar100_augmented]
# The name of the script is myjob
#SBATCH -J DoubleCNN
# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 48:00:00 # tune the number of hours ~6h*nb_models
# set the project to be charged for this
# The format should be edu<year>.DT2119 # Bertrand you have to change the code name to edu17.DD2424
#SBATCH -A edu17.DT2119
#[BERTRAND] SBATCH -A edu17.DT2424
# Use K80 GPUs (if not set, you might get nodes with Quadro K420 GPUs)
#SBATCH --gres=gpu:K420:2
# Standard error and standard output to files
#SBATCH -e error_file.txt
#SBATCH -o output_file.txt
# Run the executable (add possible additional dependencies here)
module load cudnn/5.1-cuda-8.0
./pdc_script/runtraining.sh $1
