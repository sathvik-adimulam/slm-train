# SLM for AI

I built a coding SLM intended for use in basic Python, data-analysis, and artificial intelligence workflows.

## Overview

The goal is to build a lightweight base model that understands Python, data-analysis and ML workflows well. Then, I plan to perform SFT on it to turn it to a lightweight, interactive AI agent that can assist with light business tasks. 

## Data Preparation

The training corpus is built from cleaned Python scripts and Jupyter notebooks, natural language, and math, mixed in an approximate **7:2:1 ratio** (code : text : math). This ratio is meant to keep the model strongly competent at code while also being proficient enough at English and math to handle future AI tasks (e.g. understanding prompts, performing math operations on datasets). Qwen 2.5 models were built using this ratio with their training datasets.

## Tokenizer

A byte-level BPE tokenizer was trained from scratch on a 3GB subset of the corpus. The subset was sampled to preserve the 7:2:1 code:text:math atio so the tokenizer's vocabulary would be representative of the dataset as a whole. 

## Model Architecture

A decoder-only transformer, built from scratch, with:

- **36 layers**, with embedding dimension **768**
- **RoPE** (Rotary Positional Embeddings) for positional encoding
- **SwiGLU** activations in the FFN
- **Flash Attention** for efficient attention computation

## Pre-Training

The learning rate schedule uses **linear warmup followed by cosine decay**, with warmup set to 2% (0.02) of the total decay steps. Total training steps are derived from the total token count and the gradient-accumulation batch size.

When training, BF16 and TF32 were used to maximize efficiency. Additionally, training was scaled across **2x NVIDIA B200 GPUs** using PyTorch **DDP** (Distributed Data Parallel), the largest GPU allocation available at the time.

## Future Work

- **GQA (Grouped Query Attention):** replace standard MHA with GQA to optimize KV-cache.
- **SFT (Supervised Fine-Tuning):** align the base model to follow data-analysis and coding instructions.
- **Inference Engine:** build a lightweight inference pipeline for the fine-tuned model.
