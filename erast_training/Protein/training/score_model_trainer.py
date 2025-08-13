import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset,load_from_disk,concatenate_datasets
import numpy as np
import torch
import argparse
def tokenize_function(examples):
    """将序列对编码为模型输入格式"""
    # 将两个序列连接起来，用特殊token分隔
    sequences = [
        seq1 +tokenizer.sep_token + seq2 
        for seq1, seq2 in zip(examples['sequence1'], examples['sequence2'])
    ]
    
    # 分词
    tokenized = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=512,  # 蛋白质序列可能很长，需平衡性能和效率
        return_tensors="pt"
    )
    
    # 添加标签
    tokenized['labels'] = torch.tensor(examples['label'])
    
    return tokenized


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for erast')
    #input
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    model_name = "ElnaggarLab/ankh-base"  # 蛋白质专用预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,  # 4分类任务
        problem_type="single_label_classification"
    )
    model.cuda()

    train_ds=load_from_disk(args.train)
    test_ds=load_from_disk(args.test)

    train_dataset = train_ds.map(
        tokenize_function,
        batched=True,
        batch_size=32,
        remove_columns=['sequence1', 'sequence2','id1','id2']  # 原始序列不再需要
    )
    eval_dataset = test_ds.map(
        tokenize_function,
        batched=True,
        batch_size=32,
        remove_columns=['sequence1', 'sequence2','id1','id2']  # 原始序列不再需要
    )


    # 6. 配置训练参数
    output_dir=args.out
    training_args = TrainingArguments(
        output_dir=output_dir,  # 输出目录
        num_train_epochs=5,  # 训练轮数
        per_device_train_batch_size=1,  # 批次大小
    
        warmup_steps=500,  # 预热步数
        weight_decay=0.01,  # 权重衰减
        logging_dir=os.path.join(output_dir,"log"),  # 日志目录
        logging_steps=20,  # 每100步记录一次
        evaluation_strategy="epoch",  # 每个epoch后评估
        save_strategy="steps",  # 每个epoch后保存模型
        save_steps=20000,
        # load_best_model_at_end=True,  # 训练结束时加载最佳模型
        
        report_to="tensorboard",  # 使用TensorBoard记录
        # fp16=True,  # 混合精度训练
        gradient_accumulation_steps=4,  # 梯度累积
    )

    # 7. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # 8. 开始训练
    trainer.train()
    # 9. 保存最佳模型
    trainer.save_model(f"{output_dir}/best_protein_cross_encoder")

