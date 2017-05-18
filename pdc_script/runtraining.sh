# usage ; runtraining.sh database_name [-augmentation]
# database_name in [cifar10, cifar100]

function train {
# usage : train database model_name conv_type filter_shape kernel_size kernel_pool_size
database=$1
model_name=$2
conv_type=$3
filter_shape=$4
kernel_size=$5
kernel_pool_size=$6
augmentation=$7

save_dir=model/save/${model_name}_${database}
log_dir=model/log/${model_name}_${database}
mkdir -p ${save_dir}
mkdir -p ${log_dir}

batch_size=10
train_epochs=150
patience=3
lr=0.001
learning_decay=0.5
keep_prob=0.5

echo "Run the" ${model_name} "training on the database" ${database} "..."

python src/train.py --dataset ${database} --save_dir ${save_dir} \
 --model_file model --log --path_log ${log_dir} --conv_type ${conv_type} --batch_size ${batch_size} --train_epochs ${train_epochs} \
 --patience ${patience} -lr ${lr} -filter_shape ${filter_shape} -kernel_size ${kernel_size} \
 -kernel_pool_size ${kernel_pool_size} -learning_decay ${learning_decay} -keep_prob ${keep_prob} \
 ${augmentation} \
# |& tee -a ${log_dir}/logfile.txt
}

# activate virtual env
source venv/bin/activate

# name of the db
database=$1
#database=cifar10 # cifar100
if [[ $#==2 ]]
then augmentation=$2
else augmentation=''
fi

## run learning of CNN
#model_name=CNN
#conv_type=standard
##filter_shape="128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 128,3,3 2,2 128,3,3 10,3,3"
#filter_shape="10,3,3"
#kernel_size=-1
#kernel_pool_size=-1
#
#train ${database} ${model_name} ${conv_type} "${filter_shape}" ${kernel_size} ${kernel_pool_size} ${augmentation}


## run learning of DCNN

model_name=DCNN
conv_type=double
filter_shape="128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 128,4,4 128,4,4 2,2 128,4,4 10,4,4"
filter_shape="10,4,4"
kernel_size=3
kernel_pool_size=2

train ${database} ${model_name} ${conv_type} "${filter_shape}" ${kernel_size} ${kernel_pool_size} ${augmentation}


# deactivate virtual env
deactivate
