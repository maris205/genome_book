#!/bin/sh
torchrun --nproc_per_node=8 gpt2_gene_eng_pretrain.py
