import os
import argparse
from transformers import AutoModel,AutoTokenizer,T5Tokenizer,AutoConfig
import torch
from tqdm import tqdm
from models import CaduceusForSequenceClassification
import numpy as np
import re,collections
def load_model(model_path,mode):
    if mode=="p":
        model=AutoModel.from_pretrained(model_path)
        tokenizer=AutoTokenizer.from_pretrained(model_path)
    else:
        cfg=AutoConfig.from_pretrained(model_path,trust_remote_code=True)
        model=CaduceusForSequenceClassification(cfg,num_labels=34)
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        state_dict = torch.load(os.path.join(model_path,"mamba.bin"))
        new_state_dict = collections.OrderedDict([(k.replace("module.",""), v) for k, v in state_dict.items()])
        model.load_state_dict(new_state_dict)
        model.score.cuda()
    model.cuda()
    model.eval()
    return model,tokenizer
def get_seqs(fa_path):
    sequences = []  
    current_sequence = []  
    with open(fa_path, 'r') as fasta_file:
        for line in fasta_file:
            line = line.strip()  
            if line.startswith('>'):
               
                if current_sequence:
                    sequences.append(''.join(current_sequence))
                    current_sequence = []
                continue      
            if line:  
                current_sequence.append(line)
    if current_sequence:
        sequences.append(''.join(current_sequence))  
    return sequences
@torch.no_grad()
def infer_seq(model,tokenizer,seq,mode):
    #shape:[1,dim]
    if mode=="p":
        input= tokenizer.batch_encode_plus([seq], 
                                    add_special_tokens=True, 
                                    padding='max_length',
                                    truncation=True,max_length=512,
                                    return_tensors="pt")
        input={k:v.to(model.device) for k,v in input.items()}

        output = model(input_ids=input['input_ids'],attention_mask=input['attention_mask'])
        return torch.mean(output.last_hidden_state[0],dim=0).unsqueeze(0)
    else:
        inputs=tokenizer([seq],padding=True,max_length=131072,truncation=True, return_tensors="pt")
        pooled_hidden_states, pred_label=model.inference(inputs["input_ids"].to(model.device))
        return pooled_hidden_states[0].unsqueeze(0)
def get_embs(model,tokenizer,seqs):
    emb=[]
    for seq in tqdm(seqs,desc="generate embedding"):
        emb.append(infer_seq(model,tokenizer,seq))
    return torch.cat(emb,dim=0).cpu().detach().numpy()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for erast')
    #input
    parser.add_argument('--res_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--fa_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default="p")
    args = parser.parse_args()
    model_path=args.model_path
    out_path=args.res_path
    fa_path=args.fa_path
    
    model_name=model_path.split("/")[-1]
    model,tokenizer=load_model(model_path,args.mode)
    seqs=get_seqs(fa_path)
    emb=get_embs(model,tokenizer,seqs,args.mode)
    np.save(out_path,emb)
    