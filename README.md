# 📄 LLaMA Fine-Tuning — Legal Document Summarization PoC

> LoRA fine-tuning pipeline for a LLaMA-architecture model, recreating  
> professional legal document summarization work with full experimental documentation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20%2F%20PEFT-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-PoC%20%2B%20Documented%20Findings-blueviolet?style=flat-square)

---

## 📌 Overview

This project demonstrates **actual model fine-tuning** — not just calling an AI API, but modifying a language model's internal weights — recreating the legal document summarization work originally built professionally with LLaMA at AIM.

It uses **TinyLlama-1.1B**, an open LLaMA-architecture model, specifically chosen to run entirely free on Google Colab's GPU, with no gated access and no cost. This is an intentional, documented tradeoff against the full LLaMA-7B used in production.

**What makes this different from API-based projects** (see the [AT&S Assistant](https://github.com/sanusi009) and [AVL RAG Pipeline](https://github.com/sanusi009) in this profile): this project **changes the model itself** through training, rather than prompting a fixed, unmodified model.

---

## 🔬 Why This Project Matters

| | API-based projects (AT&S, AVL) | This Project (LLaMA fine-tuning) |
|---|---|---|
| **Model behaviour** | Fixed — same model every call | **Modified** — weights actually updated |
| **Where it runs** | Cloud provider's servers | **Your own GPU** |
| **Control mechanism** | Prompts + retrieved context | **Training examples + gradient updates** |
| **Analogy** | Calling a consultant | **Sending an employee to a training course** |

---

## 🏗️ Pipeline

```
Legal Document/Summary Pairs (12 examples)
              │
              ▼
   ┌─────────────────────┐
   │  Chat Template        │   tokenizer.apply_chat_template()
   │  Formatting           │   — uses model's NATIVE format, not hand-written tags
   └──────────┬───────────┘
              ▼
   ┌─────────────────────┐
   │  LoRA Configuration   │   Low-Rank Adaptation — trains ~1% of parameters
   │  (r=16, all layers)   │   instead of the full 1.1B, fits on free GPU
   └──────────┬───────────┘
              ▼
   ┌─────────────────────┐
   │  Fine-Tuning           │   5 epochs, learning rate 8e-5
   │  (SFTTrainer)          │   bf16 precision, gradient clipping
   └──────────┬───────────┘
              ▼
   ┌─────────────────────┐
   │  Evaluation on        │   Tests on an UNSEEN legal document
   │  Held-Out Document    │   Before/after comparison vs base model
   └──────────┬───────────┘
              ▼
   ┌─────────────────────┐
   │  Findings &            │   Documents capacity limits and failure modes
   │  Documented Conclusion │   across 5 systematic experiment configurations
   └─────────────────────┘
```

---

## 🧪 Experimental Findings

This PoC systematically tested 5 different LoRA configurations:

| Configuration | LoRA Rank | Target Modules | Learning Rate | Result |
|---|---|---|---|---|
| Baseline | r=8 | q,v only | 2e-4 | Token collapse |
| Conservative | r=8 | q,v only | 5e-5 | Generic/off-topic output |
| Expanded dataset | r=8 | q,v only | 1.5e-4 | Generic/off-topic output |
| Wide LoRA, high LR | r=16 | All 7 modules | 1e-4 | Token collapse |
| Wide LoRA, low LR | r=16 | All 7 modules | 8e-5 | Mixed/partial collapse |

**Key finding:** TinyLlama-1.1B's limited capacity, combined with a small (12-example) training set, consistently produces either underfitting (generic text) or overfitting (token collapse) — confirming a known constraint in small-model LoRA fine-tuning. Production-quality results require either a larger base model (7B+), a substantially larger dataset (100s–1000s of examples), or full fine-tuning instead of LoRA.

This mirrors real production trade-offs — at AIM, the full LLaMA-7B model with a much larger curated dataset was used precisely to avoid these capacity constraints.

---

## 🐛 Debugging Journey

Building this PoC surfaced and resolved a chain of real infrastructure issues, each isolated and fixed independently:

1. **`bitsandbytes` 4-bit quantization corruption** — diagnosed via a base-model sanity check showing the issue existed *before* any fine-tuning
2. **`torchao` version incompatibility** with `peft`
3. **`trl` API changes** — `dataset_text_field`/`max_seq_length` moved from `TrainingArguments` into `SFTConfig`, and renamed to `max_length`
4. **Chat template verification** — confirmed via `tokenizer.apply_chat_template()` rather than assumed
5. **Hyperparameter tuning** across LoRA rank, target modules, and learning rate to characterize the underfitting/overfitting boundary

Each step was isolated with a dedicated sanity check before moving to the next layer of the pipeline — a debugging approach that separates infrastructure faults from genuine model/data issues.

---

## 🚀 Quick Start — Google Colab

1. Open `llama_finetune_colab_v2.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

No API keys needed — TinyLlama is fully open-access.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base model | TinyLlama-1.1B-Chat (LLaMA architecture) |
| Fine-tuning | LoRA via Hugging Face PEFT |
| Training loop | TRL `SFTTrainer` |
| Precision | bfloat16 (no quantization) |
| Hardware | Free Colab T4 GPU |
| Cost | **$0** |

---

## 🔮 Path to Production

- [ ] Scale to full LLaMA-7B/13B for sufficient model capacity
- [ ] Expand training set to 500+ real legal document/summary pairs
- [ ] Add quantitative evaluation (ROUGE, BLEU) instead of qualitative inspection
- [ ] Compare LoRA vs full fine-tuning vs QLoRA at larger scale
- [ ] Implement early stopping based on held-out validation loss

---

## 👤 Author

**Sanusi Isiaka Olatunji**  
M.Sc. Data Science — University of Leoben, Austria  
[LinkedIn](https://linkedin.com/in/sanusi-olatunji-43990198) · [GitHub](https://github.com/sanusi009)

*Recreates fine-tuning methodology from professional work at AIM (legal document summarization with LLaMA), adapted for free-tier compute with full experimental documentation.*
