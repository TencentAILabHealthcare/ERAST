import os
import argparse
from transformers import AutoModel,AutoTokenizer,T5Tokenizer
import torch
from tqdm import tqdm

import numpy as np
import re
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
def infer_seq(model,tokenizer,seq):
    #shape:[1,dim]
   
    input= tokenizer.batch_encode_plus([seq], 
                                add_special_tokens=True, 
                                padding='max_length',
                                truncation=True,max_length=512,
                                return_tensors="pt")
    input={k:v.to(model.device) for k,v in input.items()}

    output = model(input_ids=input['input_ids'],attention_mask=input['attention_mask'])
    return torch.mean(output.last_hidden_state[0],dim=0).unsqueeze(0)
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
    args = parser.parse_args()
    model_path=args.model_path
    out_path=args.res_path
    fa_path=args.fa_path
    model_name=model_path.split("/")[-1]
    model=AutoModel.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model.cuda()
    model.eval()
    seqs=get_seqs(fa_path)
    emb=get_embs(model,tokenizer,seqs)
    # outfile=f"/mnt/data8/jyn/code/erast/ood_test/swiss_ood/embs/test.npy"
    np.save(out_path,emb)
    