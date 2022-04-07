from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import logging


class MSMARCODataset(Dataset):
    def __init__(self, tokenizer:AutoTokenizer=None,queries_file_path:str=None, corpus_file_path:str=None,qrels_file_path:str=None):
        self.tokenizer=tokenizer
        self.queries,self.corpus = self.getdata(queries_file_path,corpus_file_path,qrels_file_path)
        self.queries_ids = list(self.queries.keys())

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]

        query_text = query['query']

        pos_id = query['pos'].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        tokenized_query=self.tokenizer(query_text, padding="max_length", truncation=True, max_length=300)
        tokenized_pos_doc=self.tokenizer(pos_text, padding="max_length", truncation=True, max_length=300)
        tokenized_neg_doc=self.tokenizer(neg_text, padding="max_length", truncation=True, max_length=300)

        return (tokenized_query, tokenized_pos_doc, tokenized_neg_doc)

    def __len__(self):

        return len(self.queries)

    def collate(self,batch):

        batch_query_input_ids = torch.tensor([item[0]['input_ids'] for item in batch])
        batch_query_attention_mask = torch.tensor([item[0]['attention_mask'] for item in batch])
        batch_query={
            'batch_query_input_ids':batch_query_input_ids,
            'batch_query_attention_mask':batch_query_attention_mask
        }

        batch_pos_doc_input_ids = torch.tensor([item[1]['input_ids'] for item in batch])
        batch_pos_doc_attention_mask = torch.tensor([item[1]['attention_mask'] for item in batch])
        batch_pos_doc={
            'batch_pos_doc_input_ids':batch_pos_doc_input_ids,
            'batch_pos_doc_attention_mask':batch_pos_doc_attention_mask
        }

        batch_neg_doc_input_ids = torch.tensor([item[2]['input_ids'] for item in batch])
        batch_neg_doc_attention_mask = torch.tensor([item[2]['attention_mask'] for item in batch])
        batch_neg_doc={
            'batch_neg_doc_input_ids':batch_neg_doc_input_ids,
            'batch_neg_doc_attention_mask':batch_neg_doc_attention_mask
        }
        return (batch_query,batch_pos_doc,batch_neg_doc)

    def getdata(queries_filepath,collection_filepath,qrels_filepath):

        corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
        queries = {}  

        with open(qrels_filepath) as fIn:
            for line in fIn:
                qid, pos_pids, neg_pids = line.strip('\n').split('\t')
                queries[qid]={'qid':qid,'pos':eval(pos_pids),'neg':eval(neg_pids)}

        logging.info("Read queries: {}".format(queries_filepath))
        with open(queries_filepath, 'r', encoding='utf8') as fIn:
            for idx, line in enumerate(fIn):
                qid, query = line.strip().split("\t")
                if qid not in queries:
                    continue
                query="Query: "+query
                if idx == 0:
                    logging.info(f"Train Query Example: {query}")
                queries[qid]['query'] = query

        logging.info("Read corpus: {}".format(collection_filepath))
        with open(collection_filepath, encoding='utf8') as fIn:
            for line in fIn:
                pid,title,body =line.strip().split('\t')
                title,body=title.strip(),body.strip()
                passage="Title: "+title+"Passage: "+body
                corpus[pid] = passage

        return queries,corpus



