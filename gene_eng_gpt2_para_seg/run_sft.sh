#!/bin/sh
torchrun --nproc_per_node=8 gpt2_para_seg_ft.py
