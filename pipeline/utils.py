import json
import argparse
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader
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
class SequencePairDataset(Dataset):
    """自定义数据集类用于处理序列对"""
    def __init__(self, query_seq, candidate_seqs, tokenizer, max_length=512):
        self.query_seq = query_seq
        self.candidate_seqs = candidate_seqs
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.candidate_seqs)
    
    def __getitem__(self, idx):
        candidate_seq = self.candidate_seqs[idx]
        sequence = self.query_seq + self.tokenizer.sep_token + candidate_seq
        return sequence
def collate_fn(batch, tokenizer, max_length=512):
    tokenized = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenized

def get_id_dict(fa_path):
    ids = []
    with open(fa_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):  # 识别序列头行
                # 去掉开头的'>'，分割第一个空白前的部分作为ID
                header = line[1:].strip()
                if header:  # 确保头行非空
                    # 取第一个空白（空格/制表符）前的部分作为ID
                  
                    ids.append(header)
    id_dict={}
    for i,id in enumerate(ids):
        id_dict[id]=i
    return id_dict,ids
def get_same_label(q_id:str,id_dict,pfam_dict):
    return [id_dict[item[0]] for item in pfam_dict[q_id]]
def filter_sort(original_list, filter_list):
    filter_set = set(filter_list)
    
    in_filter = []
    not_in_filter = []
  
    for item in original_list:
        if item in filter_set:
            in_filter.append(item)
        else:
            not_in_filter.append(item)
    
    return in_filter + not_in_filter,len(in_filter)
def get_filter_res(filter_res_dict,indices,q_id,t_id_dict):
    filtered_list=get_same_label(q_id,t_id_dict,filter_res_dict)
    reranker_list,filter_length=filter_sort(indices,filtered_list)
    return reranker_list,filter_length
def get_search_result(query_pfam_result, target_pfam_result):
  
    protein_pair_score_dict = {}
    for protein in query_pfam_result:
        protein_pair_score_dict[protein] = []

    for query_protein in query_pfam_result:
        for target_protein in target_pfam_result:
            if ((len(query_pfam_result[query_protein])>0) and (len(query_pfam_result[query_protein] & target_pfam_result[target_protein])>0)):
                score = 0
                protein_pair_score_dict[query_protein].append((target_protein, score))
    
    return protein_pair_score_dict
def get_pfam_res(res_path):
    data=json.load(open(res_path))
    pfam_family_output = {}
    pfam_clan_output = {}
    for prot in data:
        pfam_family_output[prot] = set()
        pfam_clan_output[prot] = set()
        for res_dict in data[prot]:
            pfam=res_dict["hmm_acc"]
            if pfam is not None and len(pfam)>0:
                pfam=pfam.split(".")[0]
                pfam_family_output[prot].add(pfam)
            pfam_clan_output[prot].add(res_dict["clan"])
    return pfam_family_output,pfam_clan_output
def filter(clan_search_result,pfam_search_result,indices,q_ids,t_id_dict):
    filter_res=[]
    lengths=[]
    for idx,q_id in enumerate(q_ids):
        all_indices,clan_length=get_filter_res(clan_search_result,indices[idx].tolist(),q_id,t_id_dict)                     
        clan_indices=all_indices[:clan_length]
        clan_indices,pfam_lenth=get_filter_res(pfam_search_result,clan_indices,q_id,t_id_dict)
        new_indices=clan_indices+all_indices[clan_length:]
        filter_res.append(new_indices)
        lengths.append(clan_length)
    return filter_res,lengths
def rerank(cands,labels):
    label_dict={0:[],1:[],2:[],3:[]}
    for i,label in enumerate(labels):
        label=label.item()
        label_dict[label].append(i)
    reranked_indices=[cands[i] for i in label_dict[3]] + [cands[i] for i in label_dict[2]] +[cands[i] for i in label_dict[1]]+[cands[i] for i in label_dict[0]]
    return reranked_indices
@torch.no_grad()
def infer(seq1,seq2,model,tokenizer):
    seq=seq1+tokenizer.sep_token+seq2
    tokenized = tokenizer(
        seq,
        padding="max_length",
        truncation=True,
        max_length=512,  # 蛋白质序列可能很长，需平衡性能和效率
        return_tensors="pt"
    )
    tokenized={k:v.to(model.device) for k,v in tokenized.items()}
    out=model(input_ids=tokenized["input_ids"],attention_mask=tokenized["attention_mask"])
    preds=out.logits.argmax(dim=-1).cpu()
    return preds
@torch.no_grad()
def infer_batch(model,tokenizer,query_seq, candidate_seqs, batch_size=32):
    # 创建数据集和数据加载器
    dataset = SequencePairDataset(query_seq, candidate_seqs, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        shuffle=False,
        num_workers=4
    )
    
    all_preds = []
    
    for batch in dataloader:
        # 将数据移至模型所在设备
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # 模型推理
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        
        all_preds.append(preds)
    
    return np.concatenate(all_preds, axis=0)
def rerank_batch(res,qs,ts,lengths,model,tokenizer):
    new_res=[]
    for i,q in enumerate(qs):
        cands=res[i][lengths[i]:]
        t_seqs=[ts[j] for j in cands]
        labels=infer_batch(model,tokenizer,q, t_seqs)
        indices=rerank(cands,labels)
        new_indices=res[i][:1]+res[i][1:lengths[i]]+indices
        new_res.append(new_indices)
    return new_res
        
        

