from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def train_tokenizer_from_file(input_file_list, vocab_size=50257, output_file="tokenizer.json"):
    """
    从文本文件训练分词器。

    参数:
    - input_file (str): 输入文本文件路径。
    - vocab_size (int): 分词器的词汇表大小。
    - output_file (str): 训练好的分词器保存路径。
    """
    # 初始化分词器
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    # 定义 BPE 训练器
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<pad>", "<unk>"]
    )

    # 开始训练
    tokenizer.train(input_file_list, trainer)
    tokenizer.save(output_file)
    print(f"分词器已保存到 {output_file}")

# 示例：从生成的文本文件训练分词器
train_tokenizer_from_file(["protein_4g.txt","dna_4g.txt","eng_4g.txt"], vocab_size=100000, output_file="gpt2_gene_eng_tokenizer.json")

