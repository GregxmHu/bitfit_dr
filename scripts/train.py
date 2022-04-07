import argparse
import gzip
import json
import logging
import os
import pickle
import random
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
import time
#### Just some code to print debug information to stdout

parser = argparse.ArgumentParser()
                    # these arguments are used for global path locating
parser.add_argument("--identifier", default=None, type=str,
                    help="to identify models")
parser.add_argument("--data_folder", default=None, type=str,
                    help="to load data files")
parser.add_argument("--checkpoint_save_folder", default=None, type=str,
                    help="to save checkpoint")

parser.add_argument("--corpus_name", default="collection.tsv", type=str)
parser.add_argument("--train_queries_name", default="queries.tsv", type=str)
parser.add_argument("--train_qrels_name", default="qrels.tsv", type=str)

parser.add_argument("--train_batch_size", default=64, type=int)

parser.add_argument("--pretrained_model_name_or_path", required=True,type=str,default=None)
parser.add_argument("--backbone_state_dict_path", required=True,type=str,default=None)
parser.add_argument("--bias_state_dict_path", required=True,type=str,default=None)

parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--pooling", default="mean",type=str,
                    help="pooling mode")
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument("--freezenonbias", action="store_true", help="Freeze all except biases in transformer")

parser.add_argument("--log_dir", type=str,default=None)

parser.add_argument("--epochs", default=10, type=int)

parser.add_argument("--warmup_steps", default=1000, type=int)

parser.add_argument("--lr", default=2e-5, type=float)

parser.add_argument("--seed", default=13, type=int)
parser.add_argument("--use_amp", action="store_true")

parser.add_argument("--local_rank", type=int, default=-1)

parser.add_argument("--round_idx", default=0,type=int)

args = parser.parse_args()


data_folder = args.data_folder

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# prepare for train
train_batch_size = args.train_batch_size  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory
num_epochs = args.epochs  # Number of epochs we want to train
optimizer_params={'lr': args.lr}
warmup_steps=args.warmup_steps
use_amp=args.use_amp
checkpoint_save_folder=args.checkpoint_save_folder
optimizer_class=AdamW
# Train the model
if args.log_dir is not None and accelerator.is_main_process:
    writer = SummaryWriter(os.path.join(args.log_dir,"round{}-".format(args.round_idx)))
    tb = writer
else:
    tb=None

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()


### load models
pretrained_model_name_or_path = args.pretrained_model_name_or_path
if args.round_idx!=0:
    backbone_state_dict_path=os.path.join(
        args.checkpoint_save_folder,"round{}-backbone_state_dict.bin".format(args.round_idx-1)
    )
    if args.freezenonbias:
        bias_state_dict_path=os.path.join(
            args.checkpoint_save_folder,"round{}-bias_state_dict.bin".format(args.round_idx-1)
        )


logging.info(args)

if args.round_idx==0:
    # first train
    model=DRT5(
        pretrain_model_path_or_name=pretrained_model_name_or_path,
        pooling_mode=args.pooling
        )
    if args.freezenonbias:
        model.load_bias()
elif args.round_idx >0:
    model=DRT5(
        pretrain_model_path_or_name=pretrained_model_name_or_path,
        pooling_mode=args.pooling,
        backbone_state_dict_path=backbone_state_dict_path
    )
    if args.freezenonbias:
        model.load_bias(
            bias_state_dict_path=bias_state_dict_path
            )

if args.freeze or args.freezenonbias:
    for name, param in model.named_parameters():
        if args.freezenonbias and name in model.bias_keys:
            # Freeze all except bias
            continue 
        param.requires_grad = False


tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
qrels_dir=os.path.join(data_folder,args.identifier)
corpus_file_path = os.path.join(data_folder, args.corpus_name)
train_queries_filepath = os.path.join(data_folder, args.train_queries_name)
if args.round_idx==0:
    train_qrels_filepath=os.path.join(data_folder, args.train_qrels_name)
else:
    train_qrels_filepath=os.path.join(qrels_dir, "round{}-".format(args.round_idx-1)+args.train_qrels_name)

train_dataset=MSMARCODataset(
    tokenizer=tokenizer,
    queries_filepath = os.path.join(data_folder, args.train_queries_name),
    qrels_file_path=train_qrels_filepath,
    corpus_file_path=corpus_file_path,
    accelerator=accelerator
    )

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataloader = accelerator.prepare(
    DataLoader(
    train_dataset, shuffle=True, batch_size=train_batch_size,
    collate_fn=train_dataset.collate
    )
)

###update optimizers
weight_decay=0.01
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
training_steps=num_epochs*len(train_dataloader)
optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
scheduler=get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
    )
optimizer=accelerator.prepare(optimizer)

### training loop

global_step = 0
skip_scheduler = False
for epoch in trange(num_epochs, desc="Epoch", disable=not accelerator.is_main_process):
    





