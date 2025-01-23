# import subprocess
# import os
# import json

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, DataCollatorWithPadding
import math

# 1. 加载 WikiText 数据集
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# 加载 GPT-2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("dnagpt/gene_eng_gpt2_v1_ft")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<p_end>"]})

max_length = 256

# 数据预处理 - 添加段落标记并过滤空文本
def add_paragraph_end_marker(example):
    text = example["text"]
    if text:  # 如果文本非空
        text = text.replace(".", "") #去掉句号防止干扰
        text = text.replace("\n", "<p_end>").strip() 
        return {"text": text}
    return None  # 返回 None 表示过滤该样本

# 应用预处理并过滤
data_with_markers = dataset.map(add_paragraph_end_marker).filter(lambda x: x is not None)

# 数据预处理
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],  # 确保字段名与数据集一致
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        if len(input_ids) < max_length:  # 填充不足部分
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        input_batch.append(input_ids)
    return {"input_ids": input_batch}


# 分词数据集
tokenized_dataset = data_with_markers.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=2)


# 5. 数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言建模
)

# 6. 加载 GPT-2 模型，并调整词表大小
model = GPT2LMHeadModel.from_pretrained("dnagpt/gene_eng_gpt2_v1_ft")
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# 7. 训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-paragraph-segmentation",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=2000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=2000,
    logging_dir="./logs",
    logging_steps=5000,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # 启用混合精度训练
    deepspeed="ds_zero2_no_offload.json"
)

# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 9. 开始训练
trainer.train()

# 10. 保存模型和分词器
trainer.save_model("./gpt2-para-seg-ft")
tokenizer.save_pretrained("./gpt2-para-seg-ft")

# 11. 评估模型 - 计算 perplexity
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")