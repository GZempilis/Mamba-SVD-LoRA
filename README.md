# Singular Value Decomposition (SVD) on Mamba Architecture with POS Tags

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GZempilis/Mamba-SVD-LoRA/blob/main/mamba_svd_lora.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Mamba](https://img.shields.io/badge/Mamba-SSM-orange.svg)

## Project Overview
State Space Models (SSMs) like **Mamba** offer a highly efficient alternative to traditional Transformer architectures by mitigating the quadratic bottleneck of global attention ($O(L)$ scaling). 

This project investigates the post-training compression of the **Mamba-130M** model using **Singular Value Decomposition (SVD)** coupled with **Low-Rank Adaptation (LoRA)**. By systematically reducing the rank of the `in_proj` and `out_proj` linear layers, we explore the critical decoupling of syntactic proficiency and semantic capacity.

## Key Findings & "The Perplexity Illusion"
* **Syntax vs. Semantics Decoupling:** SVD compression preserves the geometric skeleton required for syntax, but severely penalizes the high-resolution vector space needed for rich semantic vocabulary.
* **The Perplexity Illusion:** While introducing Part-of-Speech (POS) tags during fine-tuning dramatically drops the statistical perplexity (e.g., from 348.59 down to 4.44 in Rank 64), qualitative inference reveals a complete semantic collapse into repetition loops and glitch tokens. Statistical recovery does not equal semantic recovery!

## Tech Stack
* **Core:** Python, PyTorch, Hugging Face Transformers, PEFT (LoRA)
* **Architecture:** State Space Models (Mamba-130M)
* **Linguistic Analysis:** spaCy (POS Tagging)

## How to Run
You can run this project instantly in your browser using Google Colab by clicking the badge above[cite: 3], or run it locally by installing the dependencies:
```bash
pip install -r requirements.txt
