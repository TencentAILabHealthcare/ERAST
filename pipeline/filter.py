import json
import argparse
from tqdm import tqdm
import subprocess
PFAM_DB="/mnt/data8/jyn/code/erast/tool/pfamdb"
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
    for idx,q_id in enumerate(q_ids):
        all_indices,clan_length=get_filter_res(clan_search_result,indices[idx].tolist(),q_id,t_id_dict)                     
        clan_indices=all_indices[:clan_length]
        clan_indices,pfam_lenth=get_filter_res(pfam_search_result,clan_indices,q_id,t_id_dict)
        new_indices=clan_indices+all_indices[clan_length:]
        filter_res.append(new_indices)
    return filter_res

