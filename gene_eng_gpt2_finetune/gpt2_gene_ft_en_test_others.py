import subprocess
import os
import json

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value



from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
import evaluate
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
import json
import random


# 假设你的JSON文件名为 'data.json' 并且每行是一个独立的JSON对象
#raw_datasets = load_dataset('json', data_files='paws-x-multi-pair.jsonl')['train'].train_test_split(test_size=0.05) #默认已经shuffle
raw_datasets = load_dataset('paws-x', 'en')  # 或者指定特定语言如 'zh' 表示中文,https://huggingface.co/datasets/google-research-datasets/paws-x


#分词器
tokenizer = AutoTokenizer.from_pretrained("dnagpt/gene_eng_gpt2_v1")
tokenizer.pad_token = tokenizer.eos_token

# 修改分词器的填充方向为左侧，默认有右侧，分类问题建议左侧
#tokenizer.padding_side = "left"


#分词函数
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True,max_length=256, padding="max_length")

#构建分词后的数据集
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

#训练数据构建
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#指标函数定义
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions==labels).sum() / len(labels)}



# 动态生成随机种子
seed = random.randint(0, 10000)
#print(f"Generated seed: {seed}")
result = {}
result["seed"] = seed

training_args = TrainingArguments(
    output_dir="ds_job_dna_2222",
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    optim='adamw_torch',
    weight_decay=0.0,
    seed=seed,  # 使用动态生成的随机种子
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    num_train_epochs=4, #训练多少轮
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True
)

#模型定义，文本分类模型
model = AutoModelForSequenceClassification.from_pretrained("dnagpt/gene_eng_gpt2_v1", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train() #模型训练


def tokenize_function(example):
    #return tokenizer(example["sentence1"], example["sentence2"], truncation=True,max_length=256)
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True,max_length=256, padding="max_length")


#模型测试，英文数据集
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["en"] = ret


#模型测试，法文数据集
raw_datasets_fr = load_dataset('paws-x', 'fr')  # 或者指定特定语言如 'zh' 表示中文,https://huggingface.co/datasets/google-research-datasets/paws-x
tokenized_datasets_fr = raw_datasets_fr.map(tokenize_function, batched=True)

predictions = trainer.predict(tokenized_datasets_fr["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["fr"] = ret

#模型测试，德文数据集
raw_datasets_de = load_dataset('google-research-datasets/paws-x', 'de')  # 或者指定特定语言如 'zh' 表示中文,https://huggingface.co/datasets/google-research-datasets/paws-
tokenized_datasets_de = raw_datasets_de.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_de["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["de"] = ret

#模型测试，中文数据集
raw_datasets_zh = load_dataset('google-research-datasets/paws-x', 'zh')  # 或者指定特定语言如 'zh' 表示中文,https://huggingface.co/datasets/google-research-datasets/paws-
tokenized_datasets_zh = raw_datasets_zh.map(tokenize_function, batched=True)

predictions = trainer.predict(tokenized_datasets_zh["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["zh"] = ret

#模型测试 dna数据集，150 bp长度 简单版本
raw_datasets_dna =load_dataset('dnagpt/gene_lan_transfer', 'dna_sim_pair_simple_150bp')['train'].train_test_split(test_size=0.2) #默认已经shuffle
tokenized_datasets_dna = raw_datasets_dna.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_dna["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["dna_sim_pair_simple_150bp"] = ret

#模型测试 dna数据集，150长度，复杂版本 不相似
raw_datasets_dna = load_dataset('dnagpt/gene_lan_transfer', 'dna_sim_pair_150bp')['train'].train_test_split(test_size=0.2) #默认已经shuffle
tokenized_datasets_dna= raw_datasets_dna.map(tokenize_function, batched=True)

predictions = trainer.predict(tokenized_datasets_dna["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["dna_sim_pair_150bp"] = ret

#模型测试 dna数据集，50长度，复杂版本 不相似
raw_datasets_dna = load_dataset('dnagpt/gene_lan_transfer', 'dna_sim_pair_50bp')['train'].train_test_split(test_size=0.1) #默认已经shuffle
tokenized_datasets_dna = raw_datasets_dna.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_dna["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["dna_sim_pair_50bp"] = ret

#模型测试 蛋白质数据集，50长度/150bp，复杂版本 不相似
raw_datasets_dna_protein = load_dataset('dnagpt/gene_lan_transfer', 'protein_sim_pair_150bp')['train'].train_test_split(test_size=0.1) #默认已经shuffle
tokenized_datasets_dna_protein = raw_datasets_dna_protein.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_dna_protein["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["protein_sim_pair_150bp"] = ret

#模型测试 蛋白质数据集，150长度/450bp，复杂版本 不相似

raw_datasets_dna_protein = load_dataset('dnagpt/gene_lan_transfer', 'protein_sim_pair_450bp')['train'].train_test_split(test_size=0.1) #默认已经shuffle
tokenized_datasets_dna_protein = raw_datasets_dna_protein.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_dna_protein["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["protein_sim_pair_450bp"] = ret

#模型测试 dna-蛋白质序列
raw_datasets_dna_protein = load_dataset('dnagpt/gene_lan_transfer', 'dna_protein_pair')['train'].train_test_split(test_size=0.1) #默认已经shuffle

# 定义翻转标签的函数
def flip_labels(example):
    # 截取 sentence1 和 sentence2 的前 50 个字符,如果dna序列过长，bert分词会产生错误，只生成unk一个token
    example["sentence1"] = example["sentence1"]
    example["sentence2"] = example["sentence2"]
    #example['label'] = 1 - example['label']
    return example

# 应用翻转标签函数
flipped_datasets_dna_protein = raw_datasets_dna_protein.map(flip_labels, batched=False)

tokenized_datasets_dna_protein = flipped_datasets_dna_protein.map(tokenize_function, batched=True)
predictions = trainer.predict(tokenized_datasets_dna_protein["test"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
ret = metric.compute(predictions=preds, references=predictions.label_ids)
result["dna_protein_pair"] = ret

print(json.dumps(result))

