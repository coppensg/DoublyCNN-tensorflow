#!/bin/bash
# usage : rsync_DL_project.sh pdc_username
uname=$1
firstletter_uname=${uname:0:1}
project_dir=/cfs/klemming/nobackup/${firstletter_uname}/${uname}/DL_project/
pdc_dir=${uname}@t04n28.pdc.kth.se:${project_dir}
echo "Copy to "${pdc_dir}

echo 'Copy Datasets'
rsync -r Datasets ${pdc_dir}
echo 'Copy pdc_script'
rsync -r pdc_script ${pdc_dir}
echo 'Copy src'
rsync -r src ${pdc_dir}
echo 'Copy requirements_pdc.txt'
rsync requirements_pdc.txt ${pdc_dir}

# connect to tegner and create a symbolic link to the project
ln_cmd="ln -s ${project_dir}"
ssh ${uname}@tegner.pdc.kth.se "${ln_cmd}"

