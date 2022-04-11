import argparse
import gzip
import json
import logging
import os
import pickle
import random
from typing import List, Tuple, Dict, Set, Callable
import tarfile
from datetime import datetime
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from models.DRT5 import DRT5
from datasets.Msmarco_Dataset import MSMARCODataset
from torch.cuda.amp import autocast
from tqdm.autonotebook import trange
from torch.utils.tensorboard import SummaryWriter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch.cuda
import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import AdamW
import torch.distributed as dist
from transformers import AutoTokenizer
import time
from util import mismatched_sizes_all_gather,cos_sim,inference



parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", default="/data/private/huxiaomeng/sgpt/", type=str)
parser.add_argument("--test_csv_path", default="datasets/msmarco/", type=str)

parser.add_argument("--test_topk_score_path", default="topk_score/test/", type=str,
                    help="save topk scores of test_queries and test_corpus")
parser.add_argument("--train_topk_score_path", default="topk_score/test/", type=str,
                    help="save topk scores of test_queries and test_corpus")  

parser.add_argument("--corpus_name", default="collection.tsv", type=str)
parser.add_argument("--test_queries_name", default="queries.tsv", type=str)
parser.add_argument("--test_qrels_name", default="qrels.tsv", type=str)
parser.add_argument("--train_queries_name", default="queries.tsv", type=str)
parser.add_argument("--train_qrels_name", default="qrels.tsv", type=str)

parser.add_argument("--encode_batch_size", default=64, type=int,
                    help="batch to encode corpus or queries during inference")
parser.add_argument("--corpus_chunk_size", default=64, type=int,
                    help="split the corpus into several chunks to degrade the computation complexity")
parser.add_argument("--pretrained_model_name_or_path", required=True,type=str,default=None)
parser.add_argument("--backbone_state_dict_path", required=True,type=str,default=None)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--round_idx",type=int,default=0)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--checkpoint_save_folder",type=str,default=None)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

train_score_path=args.train_topk_score_path
test_score_path=args.test_topk_score_path
test_csv_path=args.test_csv_path


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

encode_batch_size=args.encode_batch_size
corpus_chunk_size=args.corpus_chunk_size

pooling=args.pooling
pretrained_model_name_or_path=args.pretrained_model_name_or_path
bias_state_dict_path=os.path.join(
    args.checkpoint_save_folder,"round{}/bias_state_dict.bin".format(args.round_idx)
)
backbone_state_dict_path=args.backbone_state_dict_path
logging.info(args)

model=DRT5(
    pretrain_model_path_or_name=pretrained_model_name_or_path,
    pooling_mode=args.pooling,
    backbone_state_dict_path=backbone_state_dict_path
)
model.load_bias(bias_state_dict_path)
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

data_folder=args.data_folder
collection_filepath = os.path.join(data_folder, args.corpus_name)
test_queries_file = os.path.join(data_folder, args.test_queries_name)
test_qrels_filepath = os.path.join(data_folder, args.test_qrels_name)
train_queries_file = os.path.join(data_folder, args.train_queries_name)
train_qrels_filepath = os.path.join(data_folder, args.train_qrels_name)

corpus = {}  # Our corpus pid => passage
test_queries = {}  # Our dev queries. qid => query
test_rel_docs = {}  # Mapping qid => set with relevant pids
train_queries = {}  # Our dev queries. qid => query
train_rel_docs = {}  # Mapping qid => set with relevant pids

### Download files if needed
with open(train_queries_file, encoding='utf8') as fIn:
    for idx,line in enumerate(fIn):
        #qid, query = line.strip('\n').split("\t")
        item=json.loads(line)
        qid,query=item['_id'],item['text']
        query="Query: "+query
        if idx == 0:
            logging.info(f"Train Query Example: {query}")
        train_queries[qid] = query

# Load which passages are relevant for which queries
with open(train_qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip('\n').split('\t')
        if qid not in train_queries:
            continue

        if qid not in train_rel_docs:
            train_rel_docs[qid] = set()
        train_rel_docs[qid].add(pid)
        
# Read passages
with open(collection_filepath, encoding='utf8') as fIn:
    for line in fIn:
        item=json.loads(line)
        pid,body,title=item['_id'],item['text'],item['title']
        passage="Title: "+title+"Passage: "+body
        corpus[pid] = passage
# Load the 6980 dev queries
print(len(corpus))
with open(test_queries_file, encoding='utf8') as fIn:
    for idx,line in enumerate(fIn):
        item=json.loads(line)
        qid,query=item['_id'],item['text']
        if idx == 0:
            logging.info(f"Train Query Example: {query}")
        test_queries[qid] = query

with open(test_qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip('\n').split('\t')
        if qid not in test_queries:
            continue

        if qid not in test_rel_docs:
            test_rel_docs[qid] = set()
        test_rel_docs[qid].add(pid)

ids=[id for id in list(train_queries.keys()) if id in train_rel_docs]
train_queries_ids=[id for id in ids if len(train_rel_docs[id])>0]
train_texts=[train_queries[id] for id in train_queries_ids]
ids=[id for id in list(test_queries.keys()) if id in test_rel_docs]
test_queries_ids=[id for id in ids if len(test_rel_docs[id])>0]
test_texts=[test_queries[id] for id in test_queries_ids]
cids=list(corpus.keys())
corpus_ids=[cids[idx] for idx in range(len(cids)) if idx % accelerator.num_processes == accelerator.process_index]
corpus_texts=[corpus[idx] for idx in corpus_ids]

model.eval()
model=accelerator.prepare(model)
inference(
    test_rel_docs=test_rel_docs,
    model=model,tokenizer=tokenizer,round=args.round_idx,accelerator=accelerator,
    train_queries=train_texts,test_queries=test_texts,corpus=corpus_texts,
    train_queries_ids=train_queries_ids,test_queries_ids=test_queries_ids,corpus_ids=corpus_ids,
    corpus_chunk_size=corpus_chunk_size,encode_batch_size=encode_batch_size,
    train_score_path=train_score_path,test_score_path=test_score_path,
    test_csv_path=test_csv_path
)
