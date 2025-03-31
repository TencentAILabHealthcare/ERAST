from itertools import combinations
import pandas as pd
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
import os
from tqdm import tqdm
import numpy as np

pdb_base="/mnt/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/datasets/SCOPe/scop40pdb/"
def get_residue(pdb):
    s = get_structure(pdb)
    chain = next(s.get_chains())
    coords, seq = get_residue_data(chain)
    return (coords, seq)
def get_tm_score(coords1, seq1,coords2, seq2):
    # coords1, seq1=get_residue(pdb1)
    # coords2, seq2=get_residue(pdb2)
    res = tm_align(coords1, coords2, seq1, seq2)
    return (res.tm_norm_chain1+res.tm_norm_chain2)/2
split="train"
data=pd.read_csv(f"/mnt/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/datasets/SCOPe/{split}.tsv",sep="\t")
save_path=f"/mnt/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/datasets/SCOPe/{split}_4_pairwise.txt"
residues=[]
for i in tqdm(data.index.values):
    pdb=os.path.join(pdb_base,data.loc[i]["id"])
    residues.append(get_residue(pdb)) 
all_comb = np.array(list(combinations(data.index, 2)))
##
for i in tqdm(range(int(32e6),len(all_comb))):
    comb=all_comb[i]
    pdb1=residues[comb[0]]
    pdb2=residues[comb[1]]
    tm_score=get_tm_score(pdb1[0],pdb1[1],pdb2[0],pdb2[1])
    with open(save_path,"a") as f:
        f.write(f"{str(comb[0])}\t{str(comb[1])}\t{str(tm_score)}\n")
