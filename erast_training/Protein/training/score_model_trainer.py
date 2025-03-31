import os
os.chdir("/owenbhe/buddy1/yenojiang/code/Tencent_AI/ERAST")
from Protein.model.embedding_model import Scorer
from utils import load_config,init_logger
# from accelerate import Accelerator
from datasets import load_from_disk,Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    T5EncoderModel,
    EarlyStoppingCallback,
     BitsAndBytesConfig,
)
import torch.nn as nn
import torch
from typing import Optional
from torch.utils.data import DataLoader
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

torch.distributed.init_process_group(backend='nccl')
# 获得当前进程使用的gpu号
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

BATCH_SIZE=2
def pair_collate_fn(batch):
        bacth_dict={}
        num_label=4
        for key in batch[0]:
            if key=="scores":
                scores=torch.tensor([t["scores"] for t in batch]).unsqueeze(-1)
                bacth_dict["labels"]= torch.zeros((len(batch),num_label)).scatter_(1, scores, 1)
            else:
                bacth_dict[key]=torch.tensor([t[key] for t in batch])
        return  bacth_dict
class PairTrainer(Trainer):
   
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        test_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=pair_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset=self.eval_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        
        test_dataloader = DataLoader(eval_dataset,batch_size=BATCH_SIZE,collate_fn=pair_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_dataset=self.eval_dataset
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, 
        num_replicas=int(os.environ['WORLD_SIZE']), 
        rank=int(os.environ['RANK'])
    )
        test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=pair_collate_fn, num_workers=8,  prefetch_factor=8,pin_memory=False, sampler=test_sampler)
        return test_dataloader
    # def compute_loss(self, model, inputs,return_outputs=False):
    #     inputs={k:v.to(device) for k, v in inputs.items()}
    #     output= model(**inputs)
    #     return (output.loss, output.logits) if return_outputs else output.loss



peft_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",  # or "all" or "lora_only"
)
model_checkpoint="/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/1D/models/models--ElnaggarLab--ankh-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
encoder=T5EncoderModel.from_pretrained(model_checkpoint,output_attentions=False)
encoder.gradient_checkpointing_enable()
encoder = prepare_model_for_kbit_training(encoder)   
encoder = get_peft_model(encoder, peft_config) 
encoder.to(device)
cfg=load_config("/owenbhe/buddy1/yenojiang/code/Tencent_AI/ERAST/Protein/configs/ankhScorer.yaml")
model=Scorer(encoder,cfg.ankhScorer)
model.score_head.to(device)

model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                output_device=local_rank)

train_ds=load_from_disk(cfg.datasets["train"])
test_ds=load_from_disk(cfg.datasets["test"])
# train_ds=test_ds
# test_ds=accelerator.prepare(test_ds)
config = {
     "lora_alpha": 1, 
        "lora_dropout": 0.5,
        "lr": 1e-05,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 20,
        "per_device_train_batch_size":BATCH_SIZE,
        "r": 2,
        "weight_decay": 0.3,
        # Add other hyperparameters as needed
    }
training_args = TrainingArguments(
        output_dir="/owenbhe/buddy1/yenojiang/code/Tencent_AI/PVDB/esm_attn/checkpoints/scorer/ankh",
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
        #  fp16=False,bf16=True, ###for t5,
        #   optim="paged_adamw_8bit", # added this in 
          label_names=["labels"],
          save_safetensors=False,
    )
trainer = PairTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        # tokenizer=tokenizer,
        data_collator=pair_collate_fn,
        # compute_metrics=compute_metrics
        # callbacks=[EarlyStoppingCallback(3, 0.0)]
    )

trainer.train()