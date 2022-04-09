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
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import AdamW
import torch.distributed as dist
import time
from util import mismatched_sizes_all_gather,cos_sim
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


parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--pooling", default="mean",type=str,
                    help="pooling mode")
parser.add_argument("--freeze", action="store_true", help="Freeze transformer")
parser.add_argument("--freezenonbias", action="store_true", help="Freeze all except biases in transformer")

parser.add_argument("--log_dir", type=str,default=None)

parser.add_argument("--epochs", default=10, type=int)

parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--lr", default=2e-5, type=float)

parser.add_argument("--seed", default=13, type=int)
parser.add_argument("--use_amp", action="store_true")

parser.add_argument("--local_rank", type=int, default=-1)

parser.add_argument("--round_idx", default=0,type=int)

parser.add_argument("--scale",default=20.0,type=float)

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
gradient_accumulation_steps=args.gradient_accumulation_steps
warmup_steps=args.warmup_steps
use_amp=args.use_amp
checkpoint_save_folder=args.checkpoint_save_folder
optimizer_class=AdamW
max_grad_norm=1.0
# Train the model
if args.log_dir is not None and accelerator.is_main_process:
    writer = SummaryWriter(os.path.join(args.log_dir,"round{}/".format(args.round_idx)))
    tb = writer
else:
    tb=None

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()


### load models
pretrained_model_name_or_path = args.pretrained_model_name_or_path
if args.round_idx!=0:
    backbone_state_dict_path=os.path.join(
        args.checkpoint_save_folder,"round{}/backbone_state_dict.bin".format(args.round_idx-1)
    )
    if args.freezenonbias:
        bias_state_dict_path=os.path.join(
            args.checkpoint_save_folder,"round{}/bias_state_dict.bin".format(args.round_idx-1)
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

d_model=model.embed_model.config.d_model
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
qrels_dir=os.path.join(data_folder,args.identifier)
corpus_file_path = os.path.join(data_folder, args.corpus_name)
train_queries_file_path = os.path.join(data_folder, args.train_queries_name)
if args.round_idx==0:
    train_qrels_file_path=os.path.join(data_folder, args.train_qrels_name)
else:
    train_qrels_file_path=os.path.join(qrels_dir, "round{}/".format(args.round_idx-1)+args.train_qrels_name)

train_dataset=MSMARCODataset(
    tokenizer=tokenizer,
    queries_file_path = train_queries_file_path,
    qrels_file_path=train_qrels_file_path,
    corpus_file_path=corpus_file_path,
    accelerator=accelerator
    )

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataloader = DataLoader(
    train_dataset, shuffle=False, batch_size=train_batch_size,
    collate_fn=train_dataset.collate,num_workers=24
    )

###update optimizers
weight_decay=0.01
param_optimizer = list(model.named_parameters())
#no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
no_decay=["hxm"]
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd == n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd == n for nd in no_decay)], 'weight_decay': 0.0}
]
training_steps=num_epochs*len(train_dataloader)
optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
scheduler=get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
    )
optimizer=accelerator.prepare(optimizer)
model=accelerator.prepare(model)
### training loop
model.zero_grad()
model.train()
skip_scheduler = False
loss_fnc=torch.nn.CrossEntropyLoss()
global_training_steps=0
training_steps=0
query_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)
pos_doc_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)
neg_doc_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)

for epoch in trange(num_epochs, desc="Epoch", disable=not accelerator.is_main_process):
    for batch_triple in tqdm(train_dataloader, desc="Training:",disable=not accelerator.is_main_process):
        training_steps+=1
        query=batch_triple[0]
        pos_doc=batch_triple[1]
        neg_doc=batch_triple[2]
        # encode queries
        if use_amp:
            with autocast():
                sub_query_batch_embeddings=model(
                    query['input_ids'].to(accelerator.device),
                    query['attention_mask'].to(accelerator.device)
                )
                # encode pos_docs
                sub_pos_doc_batch_embeddings=model(
                    pos_doc['input_ids'].to(accelerator.device),
                    pos_doc['attention_mask'].to(accelerator.device)
                )
                # encode neg_docs
                sub_neg_doc_batch_embeddings=model(
                    neg_doc['input_ids'].to(accelerator.device),
                    neg_doc['attention_mask'].to(accelerator.device)
                )
        else:
            sub_query_batch_embeddings=model(
                query['input_ids'].to(accelerator.device),
                query['attention_mask'].to(accelerator.device)
            )
            # encode pos_docs
            sub_pos_doc_batch_embeddings=model(
                pos_doc['input_ids'].to(accelerator.device),
                pos_doc['attention_mask'].to(accelerator.device)
            )
            # encode neg_docs
            sub_neg_doc_batch_embeddings=model(
                neg_doc['input_ids'].to(accelerator.device),
                neg_doc['attention_mask'].to(accelerator.device)
            )
        query_batch_embeddings=torch.cat(
            (query_batch_embeddings,
            sub_query_batch_embeddings)
        )
        pos_doc_batch_embeddings=torch.cat(
            ( pos_doc_batch_embeddings,
            sub_pos_doc_batch_embeddings)
        )
        neg_doc_batch_embeddings=torch.cat(
            ( neg_doc_batch_embeddings,
            sub_neg_doc_batch_embeddings)
        )
        if training_steps % gradient_accumulation_steps ==0:
            global_training_steps += 1
            shared_pos_doc_batch_embeddings=torch.cat(
                mismatched_sizes_all_gather(
                pos_doc_batch_embeddings
                )
            )
            shared_neg_doc_batch_embeddings=torch.cat(
                mismatched_sizes_all_gather(
                neg_doc_batch_embeddings
                )
            )
            candidates = torch.cat(
                [shared_pos_doc_batch_embeddings, 
                shared_neg_doc_batch_embeddings]
                )
            if use_amp:
                with autocast():
                    scores = cos_sim(
                        query_batch_embeddings, candidates
                        ) * args.scale
                    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=accelerator.device)\
                            + len(scores) * accelerator.process_index
                    
                    loss=loss_fnc(scores,labels)
                scale_before_step = scaler.get_scale()
                accelerator.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                skip_scheduler = scaler.get_scale() != scale_before_step
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()
            else:
                scores = cos_sim(
                    query_batch_embeddings, candidates
                    ) * args.scale
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=accelerator.device)\
                        + len(scores) * accelerator.process_index
                
                loss=loss_fnc(scores,labels)
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()
            dist.barrier()
            query_batch_embeddings=None
            pos_doc_batch_embeddings=None
            neg_doc_batch_embeddings=None
            query_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)
            pos_doc_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)
            neg_doc_batch_embeddings=torch.ones(0,d_model).to(accelerator.device)
            if tb :
                tb.add_scalar("loss",loss.item(),global_training_steps)
            dist.barrier()
checkpoint_save_path=os.path.join(
    args.checkpoint_save_folder,"round{}".format(args.round_idx)
)
if not os.path.exists(checkpoint_save_path):
    os.mkdir(checkpoint_save_path)
model.module.save(checkpoint_save_path)





















