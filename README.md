# genome_book
Human Genome Book:Words,Sentences and Paragraphs

# Abstract
Since the completion of the human genome sequencing project in 2001, significant progress has been made in areas such as gene regulation editing and protein structure prediction. However, given the vast amount of genomic data, the segments that can be fully annotated and understood remain relatively limited. If we consider the genome as a book, constructing its equivalents of words, sentences, and paragraphs has been a long-standing and popular research direction. Recently, studies on transfer learning in large language models have provided a novel approach to this challenge.Multilingual transfer ability, which assesses how well models fine-tuned on a source language can be applied to other languages, has been extensively studied in multilingual pre-trained models. Similarly, the transfer of natural language capabilities to "DNA language" has also been validated. Building upon these findings, we first trained a foundational model capable of transferring linguistic capabilities from English to DNA sequences. Using this model, we constructed a vocabulary of DNA words and mapped DNA words to their English equivalents.Subsequently, we fine-tuned this model using English datasets for paragraphing and sentence segmentation to develop models capable of segmenting DNA sequences into sentences and paragraphs. Leveraging these models, we processed the GRCh38.p14 human genome by segmenting, tokenizing, and organizing it into a "book" comprised of genomic "words," "sentences," and "paragraphs." Additionally, based on the DNA-to-English vocabulary mapping, we created an "English version" of the genomic book. This study offers a novel perspective for understanding the genome and provides exciting possibilities for developing innovative tools for DNA search, generation, and analysis.

# train data
pretrain data could be found in https://huggingface.co/dnagpt/genome_book

# DNA model list
* tokenizer BPE train
* gene_eng_gpt2_pretrain pretrained model
-- gene_eng_gpt2_finetune finetune model with paws-x
-- gene_eng_gpt2_para_seg paragraph segment model
-- gene_eng_gpt2_sentence_seg  sentence segment model
-- gene_eng_gpt2_summary  summary model
-- dna_word_trans  DNA to ENG translation

# other dir
-- book_example genome book example
-- generate_book process to generate genome book, follow step index 
