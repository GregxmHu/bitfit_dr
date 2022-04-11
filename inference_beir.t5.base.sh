set -ex
export OMP_NUM_THREADS=1

model=$1
pooling=$2
datasets=$3
pre_round=$4
round=0

test_queries_name="queries.test.jsonl"
test_qrels_name="qrels.test.tsv"
train_queries_name="queries.test.jsonl"
train_qrels_name="qrels.test.tsv"
corpus_name="corpus.jsonl"

identifier="${model}.bitfit_${pooling}-pooling_${datasets}"
pre_id="${model}.prefinetune_${pooling}-pooling_msmarco"
project_path="/data/home/scv0540/run/my_dr"
data_folder="${project_path}/datasets/bitfit/${datasets}/"
checkpoint_save_folder="${project_path}/checkpoints/${identifier}/"
pre_ckpt="${project_path}/checkpoints/${pre_id}/"
pretrained_model_name_or_path="/data/home/scv0540/run/pretrained_models/t5-base"
backbone_state_dict_path="$pre_ckpt/round$4/backbone_state_dict.bin"
test_csv_path="${project_path}/results/${identifier}/test.csv"
test_topk_score_path="${project_path}/scores/${identifier}/test.tsv"
train_topk_score_path="${project_path}/scores/${identifier}/train.tsv"

accelerate launch\
 --config_file accelerate_config.yaml\
 src/beir_inference.py\
 --checkpoint_save_folder $checkpoint_save_folder\
 --data_folder $data_folder\
 --test_csv_path $test_csv_path\
 --test_topk_score_path $test_topk_score_path\
 --train_topk_score_path $train_topk_score_path\
 --corpus_name $corpus_name\
 --test_queries_name $test_queries_name\
 --test_qrels_name $test_qrels_name\
 --train_queries_name $train_queries_name\
 --train_qrels_name $train_qrels_name\
 --encode_batch_size 300\
 --corpus_chunk_size 100000\
 --pretrained_model_name_or_path ${pretrained_model_name_or_path}\
 --backbone_state_dict_path $backbone_state_dict_path\
 --pooling $pooling\
 --round_idx $round\
 --seed 13\

