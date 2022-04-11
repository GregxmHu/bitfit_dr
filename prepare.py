import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sgpt-125M", type=str)
parser.add_argument("--pooling", default="weightedmean", type=str)
parser.add_argument("--datasets", default="sgpt-125M", type=str)
parser.add_argument("--total_round", default=2, type=int)
parser.add_argument("--suffix",default="prefinetune",type=str)
#parser.add_argument("--margin", default=0.1, type=int)
args = parser.parse_args()
## prepare qrels
qrels_dir="/data/home/scv0540/run/my_dr/datasets/{}.{}_{}-pooling_{}".format(args.model,args.suffix,args.pooling,args.datasets)
if os.path.exists(qrels_dir):
    os.system("rm -r {}".format(qrels_dir))

os.mkdir(qrels_dir)
for idx in range(args.total_round):
    dir_idx=os.path.join(qrels_dir,"round{}".format(idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)

qrels_path_list_file="{}/qrels_path.tsv".format(qrels_dir)
with open(qrels_path_list_file,'w') as f:
    pass
# prepare checkpoints
checkpoint_dir="/data/home/scv0540/run/my_dr/checkpoints/{}.{}_{}-pooling_{}".format(args.model,args.suffix,args.pooling,args.datasets)
if os.path.exists(checkpoint_dir):
    os.system("rm -r {}".format(checkpoint_dir))
os.mkdir(checkpoint_dir)
for idx in range(args.total_round):
    dir_idx=os.path.join(checkpoint_dir,"round{}".format(idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
checkpoint_path_list_file="{}/checkpoint_path.tsv".format(checkpoint_dir)
with open(checkpoint_path_list_file,'w') as f:
    pass
# prepare results files
results_dir="/data/home/scv0540/run/my_dr/results/{}.{}_{}-pooling_{}".format(args.model,args.suffix,args.pooling,args.datasets)
if os.path.exists(results_dir):
    os.system( "rm -r {}".format(results_dir))
os.mkdir(results_dir)
test_results_path=os.path.join(results_dir,"test.csv")
with open(test_results_path,'w') as f:
    pass
# prepare score files
score_dir="/data/home/scv0540/run/my_dr/scores/{}.{}_{}-pooling_{}".format(args.model,args.suffix,args.pooling,args.datasets)
if os.path.exists(score_dir):
    os.system("rm -r {}".format(score_dir))
os.mkdir(score_dir)
train_score_path=os.path.join(score_dir,"train.tsv")
test_score_path=os.path.join(score_dir,"test.tsv")
with open(train_score_path,'w') as f:
    pass
with open(test_score_path,'w') as f:
    pass
# prepare logs files
logs_dir="/data/home/scv0540/run/my_dr/logs/{}.{}_{}-pooling_{}".format(args.model,args.suffix,args.pooling,args.datasets)
if os.path.exists(logs_dir):
    os.system("rm -r {}".format(logs_dir))
os.mkdir(logs_dir)
