import os
os.chdir("/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB")
from DNAemb.model.modeling_caduceus import CaduceusForSequenceClassification

# from accelerate import Accelerator
from datasets import load_from_disk,Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch.nn as nn
import torch
from typing import Optional
from torch.utils.data import DataLoader
import pickle
torch.distributed.init_process_group(backend='nccl')
# 获得当前进程使用的gpu号
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

BATCH_SIZE=32
label_map=pickle.load(open("/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/DNAemb/datasets/distant/label_map.pkl","rb"))
def phy_collate_fn(batch):
        bacth_dict={}
        scores=torch.tensor([label_map[t["phylum"]] for t in batch]).unsqueeze(-1)
        bacth_dict["labels"]= torch.zeros((len(batch),len(label_map))).scatter_(1, scores, 1)
        bacth_dict["input_ids"]=torch.tensor([t["frag"] for t in batch])
        return  bacth_dict
class PhyTrainer(Trainer):
   
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        test_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=phy_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset=self.eval_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        
        test_dataloader = DataLoader(eval_dataset,batch_size=BATCH_SIZE,collate_fn=phy_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_dataset=self.eval_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=phy_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
   
ckpt="/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/1D/models/models--kuleshov-group--caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
cfg=AutoConfig.from_pretrained(ckpt,trust_remote_code=True)
model=CaduceusForSequenceClassification(cfg,device=device,num_labels=len(label_map))
model.score.to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                output_device=local_rank)
train_ds=load_from_disk("/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/DNAemb/datasets/distant/train")
test_ds=load_from_disk("/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/DNAemb/datasets/distant/test")
# train_ds=test_ds
# test_ds=accelerator.prepare(test_ds)
config = {
        "lr": 1e-06,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 20,
        "per_device_train_batch_size":BATCH_SIZE,
        "r": 2,
        "weight_decay": 0.3,
        # Add other hyperparameters as needed
    }
training_args = TrainingArguments(
        output_dir="/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/DNAemb/trails",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1, # changed from 1 to 4
        # warmup_steps=2, # added this in 
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        load_best_model_at_end=True,
        # metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=3,
        no_cuda=False,
        seed=8893,
        report_to="tensorboard", 
        prediction_loss_only=True,
        save_safetensors=False,
        label_names=["labels"]
    )
trainer = PhyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        # tokenizer=tokenizer,
        data_collator=phy_collate_fn,
        # compute_metrics=compute_metrics
        # callbacks=[EarlyStoppingCallback(3, 0.0)]
    )
# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

trainer.train()