from turtle import position
from transformers import AutoModel
import torch
from accelerate import Accelerator
import collections
import os
from typing import List, Tuple, Dict

class DRT5(torch.nn.Module):
    def __init__(self,pretrain_model_path_or_name:str=None,pooling_mode:str="cls",backbone_state_dict_path:str=None):
        super(DRT5,self).__init__()
        ## load pretrain model
        self.embed_model=AutoModel.from_pretrained(pretrain_model_path_or_name)
        self.scale()
        self.pooling_mode=pooling_mode
        if backbone_state_dict_path is not None:
            self.embed_model.load_state_dict(
                torch.load(backbone_state_dict_path)
            )
        self.backbone_keys=self.embed_model.state_dict().keys()

    def load_bias(self,bias_state_dict_path:str=None):
        model_state_dict=self.embed_model.state_dict()
        ### add bias module to embed_model
        if not hasattr(self,'bias_keys'):
            self.add_bias()
        ### load bias
        if bias_state_dict_path is not None:
            bias_state_dict=torch.load(bias_state_dict_path)
            model_state_dict.update(bias_state_dict)
            self.embed_model.load_state_dict(model_state_dict)

    def scale(self):
        for i in range(self.embed_model.config.num_layers):
            self.embed_model.state_dict()[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
            self.embed_model.state_dict()[f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'] /= 10
            self.embed_model.state_dict()[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] /= 10

            self.embed_model.state_dict()[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] /= 100
            self.embed_model.state_dict()[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
            self.embed_model.state_dict()[f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'] /= 10
            self.embed_model.state_dict()[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] /= 10
        self.embed_model.state_dict()['shared.weight'] /= 100

    def add_bias(self):
        self.bias_keys=[]
        for i in range(self.embed_model.config.num_layers):
        ## mark bias keys
            self.bias_keys.append("encoder.block.{}.layer.1.DenseReluDense.wi.bias".format(i))
            self.bias_keys.append("encoder.block.{}.layer.1.DenseReluDense.wo.bias".format(i))
            self.bias_keys.append("decoder.block.{}.layer.2.DenseReluDense.wi.bias".format(i))
            self.bias_keys.append("decoder.block.{}.layer.2.DenseReluDense.wo.bias".format(i))
        ## encoder
            enc_wi=self.embed_model.encoder.block[i].layer[1].DenseReluDense.wi
            enc_wo=self.embed_model.encoder.block[i].layer[1].DenseReluDense.wo
            #wi
            new_enc_wi=torch.nn.Linear(self.embed_model.config.d_model,self.embed_model.config.d_ff,bias=True)
            new_enc_wi.weight.data=enc_wi.weight.data.clone().detach()
            new_enc_wi.requires_grad_(True)
            self.embed_model.encoder.block[i].layer[1].DenseReluDense.wi=new_enc_wi
            #wo
            new_enc_wo=torch.nn.Linear(self.embed_model.config.d_ff,self.embed_model.config.d_model,bias=True)
            new_enc_wo.weight.data=enc_wo.weight.data.clone().detach()
            new_enc_wo.requires_grad_(True)
            self.embed_model.encoder.block[i].layer[1].DenseReluDense.wo=new_enc_wo
        ## decoder
            dec_wi=self.embed_model.decoder.block[i].layer[2].DenseReluDense.wi
            dec_wo=self.embed_model.decoder.block[i].layer[2].DenseReluDense.wo
            #wi
            new_dec_wi=torch.nn.Linear(self.embed_model.config.d_model,self.embed_model.config.d_ff,bias=True)
            new_dec_wi.weight.data=dec_wi.weight.data.clone().detach()
            new_dec_wi.requires_grad_(True)
            self.embed_model.decoder.block[i].layer[2].DenseReluDense.wi=new_dec_wi
            #wo
            new_dec_wo=torch.nn.Linear(self.embed_model.config.d_ff,self.embed_model.config.d_model,bias=True)
            new_dec_wo.weight.data=dec_wo.weight.data.clone().detach()
            new_dec_wo.requires_grad_(True)
            self.embed_model.decoder.block[i].layer[2].DenseReluDense.wo=new_dec_wo

    def save(self,save_path):
        st=self.embed_model.state_dict()
        if hasattr(self,'bias_keys'):
            bias_state_dict=collections.OrderedDict({k:st[k] for k in self.bias_keys})
            torch.save(bias_state_dict,os.path.join(save_path,"bias_state_dict.bin"))
        backbone_state_dict=collections.OrderedDict({k:st[k] for k in self.backbone_keys})
        torch.save(backbone_state_dict,os.path.join(save_path,"backbone_state_dict.bin"))

    def forward(self,input_ids,attention_mask):
        decoder_input_ids=torch.zeros_like(input_ids).to(input_ids.device)
        ## get embeddings
        sentence_embeddings=self.embed_model(
            decoder_input_ids=decoder_input_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )['last_hidden_state']
        ## pool embeddings
        if self.pooling_mode=="cls":
            position_weights=torch.zeros_like(sentence_embeddings).to(input_ids.device)
            position_weights[:,0:]=1

        return torch.sum(
            sentence_embeddings*position_weights,dim=1
        )        


    
