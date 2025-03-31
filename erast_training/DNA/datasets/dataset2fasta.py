import pandas as pd
ds_dir="/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/benchmarks/scope_all/df-ebd.tsv"
out_dir=ds_dir.replace(".tsv",".fa")
data=pd.read_csv(ds_dir,sep="\t",header=None,names=['id',"seq"])
with open(out_dir,"w") as outfile:
    for line in data.itertuples():
        # outfile.write(f">{line.tax} {line.Index}\n")
        id=line.id.split(",")[0]
        outfile.write(f">{id}\n")
        outfile.write(f"{line.seq}\n")