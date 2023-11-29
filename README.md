# Contlo-FT-Assignment
## GPT-2 with Advanced Architectural Modifications

This repository contains an implementation of the GPT-2 language model with advanced architectural modifications, including Rotary Positional Embedding, Group Query Attention, and Sliding Window Attention. The model is designed to be trained and evaluated in various settings: Single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP).


## Introduction

This project extends the original GPT-2 model with three architectural modifications:
1. **Rotary Positional Embedding**
2. **Group Query Attention**
3. **Sliding Window Attention**

The project also includes a flexible training loop that can be adapted for single GPU, DDP, and FSDP setups.

## Requirements

- Python 3.x
- PyTorch
- FairScale (for FSDP)
- Other dependencies (install with `pip install -r requirements.txt`)


## Results and Model Analysis

After training, evaluate the model's performance and analyze the impact of each architectural modification on tasks like language modeling or text generation. Check the `Results` directory for saved models and training logs.

## Acknowledgments

This project builds upon the original GPT-2 model and incorporates advancements proposed by various research papers. Special thanks to the authors of RoFormer, GQA, and Longformer for their insightful contributions to transformer architectures.
