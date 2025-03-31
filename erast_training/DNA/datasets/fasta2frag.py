from pandas import DataFrame
import argparse
from Bio.SeqIO import parse
import os
save_dir="/mnt/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/database/refseq/parts/"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta')
    args = parser.parse_args()
    records = list(parse(args.fasta, 'fasta'))
    file=args.fasta.split("/")[-1].replace(".fna","")
    # fragments = [str(r.seq) for r in records]
    # species_list = [r.id for r in records]
    seq = [str(r.seq) for r in records]
    ids=[str(r.id) for r in records]
    name=[str(r.name) for r in records]
    description=[str(r.description) for r in records]
    data=DataFrame({"seq":seq,"id":ids,"name":name,"description":description})
    data.to_csv(os.path.join(save_dir,f"{file}.tsv"),sep="\t",index=False)
    # prefix = save_dir+args.fasta.split("/")[-1].split(".")[0]
    # json.dump(fragments, open(prefix+ '_fragments.json', 'w'))
    
    # with open(prefix + '_species_picked.txt', 'w') as f:
    #         f.write('\n'.join(species_list))