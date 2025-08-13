from utils import get_pfam_res,get_search_result,get_id_dict,filter,get_seqs,rerank_batch
import numpy as np
import json
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
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
def similarity_search(q_matrix,t_matrix, normalize=True):
    if normalize:
       
        norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 
        q_matrix = q_matrix / norms
        norms = np.linalg.norm(t_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 
        t_matrix = t_matrix / norms
    
    similarity_matrix = np.dot(q_matrix, t_matrix.T)
    
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    sorted_scores = np.sort(similarity_matrix, axis=1)[:, ::-1]
    return sorted_indices, sorted_scores
def pipeline(out_path,pfam_q_path,pfam_t_path,emb_q_path,emb_t_path,fa_q_path,fa_t_path,reranker):
    q_emb=np.load(emb_q_path)
    t_emb=np.load(emb_t_path)
    indices, scores = similarity_search(q_emb,t_emb)
    q_pfam_dict,q_clan_dict=get_pfam_res(pfam_q_path)
    t_pfam_dict,t_clan_dict=get_pfam_res(pfam_t_path)
    clan_search_result=get_search_result(q_clan_dict,t_clan_dict)
    pfam_search_result = get_search_result(q_pfam_dict, t_pfam_dict)
    t_id_dict,t_ids=get_id_dict(fa_t_path)
    q_id_dict,q_ids=get_id_dict(fa_q_path)
    res,lengths=filter(clan_search_result,pfam_search_result,indices,q_ids,t_id_dict)
    if reranker_path is not None:
        qs=get_seqs(fa_q_path)
        ts=get_seqs(fa_t_path)
        model= AutoModelForSequenceClassification.from_pretrained(reranker_path)
        tokenizer=AutoTokenizer.from_pretrained(reranker_path)
        res=rerank_batch(res,qs,ts,lengths,model,tokenizer)
    with open(out_path,"w") as f:
        for q_idx in range(len(q_ids)):
            targets=[t_ids[t_idx] for t_idx in res[q_idx]]
            f.write(json.dumps({"query":q_ids[q_idx],"res":targets})+"\n")
            # for i,t_idx in enumerate(res[q_idx]):
            #     f.write(q_ids[q_idx]+"\t"+t_ids[t_idx]+"\t"+str(scores[q_idx][i].item())+"\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for erast')
    #input
    parser.add_argument('--fa_q', type=str, default="example/q.fa")
    parser.add_argument('--fa_t', type=str, default="example/t.fa")
    parser.add_argument('--pfam_q', type=str, default="example/pfam_q.json")
    parser.add_argument('--pfam_t', type=str, default="example/pfam_t.json")
    parser.add_argument('--emb_q', type=str, default="example/q_emb.npy")
    parser.add_argument('--emb_t', type=str, default="example/t_emb.npy")
    parser.add_argument('--reranker', type=str, default=None)
    parser.add_argument('--out', type=str, default="example/out.jsonl")
    args = parser.parse_args()
    out_path=args.out
    pfam_q_path=args.pfam_q
    pfam_t_path=args.pfam_t
    emb_q_path=args.emb_q
    emb_t_path=args.emb_t
    fa_q_path=args.fa_q
    fa_t_path=args.fa_t
    reranker_path=args.reranker
    
    pipeline(out_path,pfam_q_path,pfam_t_path,emb_q_path,emb_t_path,fa_q_path,fa_t_path,reranker_path)
    