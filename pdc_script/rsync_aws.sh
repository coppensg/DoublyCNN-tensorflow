#!/bin/bash
# usage : rsync_aws.sh ip_of_ec2_instance
ec2_ip=$1
rsync -azvv -e "ssh -i $HOME/aws_ssh/projet_dl.pem" -r src/ ubuntu@$ec2_ip:~/DL_project/src/
rsync -azvv -e "ssh -i $HOME/aws_ssh/projet_dl.pem" -r pdc_script/ ubuntu@$ec2_ip:~/DL_project/pdc_script

# get back trained models
#rsync -azvv -e "ssh -i $HOME/aws_ssh/projet_dl.pem" -r ubuntu@$ec2_ip:~/DL_project/model model_learned/
