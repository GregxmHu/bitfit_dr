set -ex
export OMP_NUM_THREADS=1
model=$1
pooling=$2
datasets=$3
round_idx=$4
identifier="${model}_${pooling}-pooling_${datasets}"
project_path="/data/home/scv0540/run/my_dr"
data_folder="${project_path}/datasets/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
corpus_name="corpus_with_title.tsv"
train_queries_name="queries.train.tsv"
train_qrels_name="qrels-irrels.train.tsv"
log_dir="$project_path/logs/$identifier"
pretrained_model_name_or_path="/data/home/scv0540/run/pretrained_models/t5-small"
#mkdir $checkpoint_save_folder
#mkdir $log_dir
accelerate launch\
 --config_file accelerate_config.yaml\
 src/train.py\
 --identifier $identifier\
 --data_folder $data_folder\
 --checkpoint_save_folder $checkpoint_save_folder\
 --corpus_name $corpus_name\
 --train_queries_name $train_queries_name\
 --train_qrels_name $train_qrels_name\
 --pretrained_model_name_or_path ${pretrained_model_name_or_path}\
 --train_batch_size 24\
 --pooling $pooling\
 --epochs 1\
 --use_amp\
 --log_dir=$log_dir\
