import requests
import torch
from torch import Tensor, device
from typing import List, Callable
from tqdm.autonotebook import tqdm
from tqdm.autonotebook import trange
import sys
import importlib
import os
import torch
import numpy as np
import queue
from models.DRT5 import DRT5
from transformers import AutoTokenizer
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
import logging
class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply


def mismatched_sizes_all_gather(tensor: Tensor, group=None, async_op=False):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    dim_0_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device="cuda")
    sizes = [torch.zeros_like(dim_0_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, dim_0_size, group=group, async_op=async_op)
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros((max_size, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
    padded[:tensor.shape[0], :] = tensor
    # gather the padded tensors
    tensor_list = [torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype) for _ in range(world_size)]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        assert not tensor_list[rank][sizes[rank]:, :].count_nonzero().is_nonzero(), \
            "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank][:sizes[rank], :]
    return tensor_list

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def encode(texts=None, 
    batch_size:int=500, model:DRT5=None,tokenizer:AutoTokenizer=None,accelerator:Accelerator=None,
    max_seq_length:int =300
    ):
    assert texts is not None
    total_embeddings=torch.ones(0,model.module.embed_model.config.d_model).to(
        accelerator.device
    )
    #dataloader=DataLoader(
    #    texts,shuffle=False,batch_size=batch_size,num_workers=6
    #)
    for texts_start_idx in trange(0,len(texts),batch_size, desc='Encode Texts',disable=not accelerator.is_main_process):
        texts_end_idx=min(len(texts),texts_start_idx+batch_size)
        sub_texts=texts[texts_start_idx:texts_end_idx]
        inputs=tokenizer(sub_texts,padding="max_length",max_length=max_seq_length,truncation=True,return_tensors="pt")
        with torch.no_grad():
            sub_embeedings=model(
                input_ids=inputs['input_ids'].to(accelerator.device),
                attention_mask=inputs['attention_mask'].to(accelerator.device),
            )
        total_embeddings=torch.cat(
            (total_embeddings,sub_embeedings)
        )
    return total_embeddings


def inference( model, train_queries,corpus,encode_batch_size,corpus_chunk_size,round,tokenizer,
    corpus_ids,train_queries_ids,accelerator,test_queries,test_queries_ids,
    train_score_path,test_score_path,test_rel_docs,
    test_csv_path) :
        name='cos_sim'
        test_csv_headers=["round","cos_sim-mrr@10","cos_sim-ndcg@10"]
        #model=accelerator.prepare(model)
        k=10
        #dist.barrier()
        if accelerator.is_main_process:
            print("encode training queries")
        #dist.barrier()
        train_query_embeddings = encode(texts=train_queries, model=model, batch_size=encode_batch_size, accelerator=accelerator,tokenizer=tokenizer)
        #dist.barrier()
        if accelerator.is_main_process:
            print("encode test queries")
        #dist.barrier()
        test_query_embeddings = encode(texts=test_queries, model=model, batch_size=encode_batch_size, accelerator=accelerator,tokenizer=tokenizer)

        test_queries_result_list = {}
        #dist.barrier()
        if accelerator.is_main_process:
            print("Train Queries: {}   Test Queries: {}".format(len(train_queries),len(test_queries)))
        #dist.barrier()
        test_queries_result_list['cos_sim'] = [[] for _ in range(len(test_query_embeddings))]
        for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size, desc='Encode Corpus',disable=not accelerator.is_main_process):
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
            batch_ids=corpus_ids[corpus_start_idx:corpus_end_idx]
            batch_texts=corpus[corpus_start_idx:corpus_end_idx]
            batch_embeddings = encode(
                texts=batch_texts,
                model=model,
                batch_size=encode_batch_size, 
                accelerator=accelerator,
                tokenizer=tokenizer
            )
            for train_query_start_idx in trange(0,len(train_queries_ids),3000,disable=True):
                train_query_end_idx = min(train_query_start_idx + 3000, len(train_queries_ids))
                train_sub_query_embeddings=train_query_embeddings[train_query_start_idx:train_query_end_idx]
                train_pair_scores = cos_sim(train_sub_query_embeddings, batch_embeddings)

                #Get top-k values
                train_pair_scores_top_k_values, train_pair_scores_top_k_idx = torch.topk(train_pair_scores, 1, dim=1, largest=True, sorted=False)
                train_pair_scores_top_k_values = train_pair_scores_top_k_values.cpu().tolist()
                train_pair_scores_top_k_idx = train_pair_scores_top_k_idx.cpu().tolist()
                with open(train_score_path,'a+') as f:
                    for train_sub_query_itr in trange(0,len(train_sub_query_embeddings),1,disable=True):
                        for id, train_score in zip(train_pair_scores_top_k_idx[train_sub_query_itr], train_pair_scores_top_k_values[train_sub_query_itr]):
                            train_query_itr=train_query_start_idx+train_sub_query_itr
                            did=batch_ids[id]
                            train_qid=train_queries_ids[train_query_itr]
                            f.write(str(train_query_itr)+'\t'+train_qid+'\t'+did+'\t'+'cos_sim'+'\t'+str(train_score)+'\n')

            test_pair_scores = cos_sim(test_query_embeddings, batch_embeddings)

            #Get top-k values
            test_pair_scores_top_k_values, test_pair_scores_top_k_idx = torch.topk(test_pair_scores, min(k, len(test_pair_scores[0])), dim=1, largest=True, sorted=False)
            test_pair_scores_top_k_values = test_pair_scores_top_k_values.cpu().tolist()
            test_pair_scores_top_k_idx = test_pair_scores_top_k_idx.cpu().tolist()
            with open(test_score_path,'a+') as f:
                for test_query_itr in trange(0,len(test_query_embeddings),1,disable=True):
                    for id, test_score in zip(test_pair_scores_top_k_idx[test_query_itr], test_pair_scores_top_k_values[test_query_itr]):
                        did=batch_ids[id]
                        test_qid=test_queries_ids[test_query_itr]
                        f.write(str(test_query_itr)+'\t'+test_qid+'\t'+did+'\t'+'cos_sim'+'\t'+str(test_score)+'\n')
            if accelerator.is_main_process:
                print("save test query-passage scores")
        print("Corpus: {}\n".format(len(corpus)))
            
        
        if accelerator.is_local_main_process:
            with open(test_score_path,'r') as f:
                for item in f:
                    pair_score=item.strip('\n').split('\t')
                    query_itr,did,name,score=eval(pair_score[0]),pair_score[2],pair_score[3],eval(pair_score[4])
                    test_queries_result_list['cos_sim'][query_itr].append(
                                {   'corpus_id': did, 'score': score }
                                )
            test_scores = {
                'cos_sim': compute_metrics(test_queries_result_list['cos_sim'],test_queries_ids,test_rel_docs,test_queries) 
                }
            

            if not os.path.isfile(test_csv_path):
                fOut = open(test_csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(test_csv_headers))
                fOut.write("\n")

            else:
                fOut = open(test_csv_path, mode="a", encoding="utf-8")

            output_data = [round]
            output_data.append(test_scores['cos_sim']['mrr@k'][k])
            output_data.append(test_scores['cos_sim']['ndcg@k'][k])
            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

    
def compute_metrics( queries_result_list,queries_ids,relevant_docs,queries):
        mrr = {10:0}
        ndcg = {10:[] }
        queries_ids=queries_ids
        relevant_docs=relevant_docs
        queries=queries
        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = queries_ids[query_itr]
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = relevant_docs[query_id]
            # MRR@k
            for k_val in [10]:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        mrr[k_val] += 1.0 / (rank + 1)
                        break
            # NDCG@k
            for k_val in [10]:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)
                ndcg_value = compute_dcg_at_k(predicted_relevance, k_val) / compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)
        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])
        for k in mrr:
            mrr[k] /= len(queries)
        return { 'ndcg@k': ndcg, 'mrr@k': mrr}

def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
    return dcg
