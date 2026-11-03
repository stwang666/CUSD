# CUSD

> **Separating Common and Unique Directions for Data-Free Model Merging**
>

## Overview

The code includes:

- **CUSD decomposition and merging** (`cusd/`): the core algorithm.
- **Baseline merging methods** (`methods/`): TA.
- **Experiment runners** (`run_vit.py`, `run_roberta.py`, `run_llm.py`): end-to-end merge + evaluation scripts.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- See `requirements.txt` for full list

```bash
pip install -r requirements.txt
```

## Model-Specific Environments

The CUSD decomposition and merging code (`cusd/`, `methods/`) has minimal dependencies
(PyTorch, NumPy, SciPy). The evaluation environments depend on the model family:

- **ViT**: please refer https://github.com/mlfoundations/task_vectors.
- **RoBERTa**: please refer https://github.com/nathanielyvo/WUDI-Merging/tree/main/nlp_roberta.
- **LLM**: please refer https://github.com/EleutherAI/lm-evaluation-harness

## Checkpoints and Evaluation

### ViT (ViT-B/32, ViT-B/16, ViT-L/14)

**Checkpoints**: We use the finetuned ViT checkpoints from
[Ilharco et al. (2023)](https://github.com/mlfoundations/task_vectors).

- Checkpoints (Google Drive): [https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)

### RoBERTa (RoBERTa-Base, RoBERTa-Large)

**Checkpoints**: We use the finetuned RoBERTa checkpoints from
[WUDI-Merging (Zhang et al., 2025)](https://github.com/nathanielyvo/WUDI-Merging/tree/main/nlp_roberta).

- Checkpoints (Google Drive): [https://drive.google.com/drive/folders/1pArQjN8f0DIdLcmv-YDoIYnKeNG0HJV2](https://drive.google.com/drive/folders/1pArQjN8f0DIdLcmv-YDoIYnKeNG0HJV2)

### LLM (Llama-3.2-3B, Llama-3.1-8B, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-2B, Gemma-2-9B)

**Checkpoints**: We use the finetuned expert models from
[MergeBench](https://huggingface.co/MergeBench).

## Usage

### ViT experiments

```bash
# Plain baseline (Task Arithmetic)
python run_vit.py --method ta --mode plain --vit-model ViT-B-16 \
    --ckpt-dir /path/to/checkpoints/ViT-B-16 \
    --eval-dir /path/to/task_vectors

# CUSD + Task Arithmetic (beta=0)
python run_vit.py --method ta --mode cusd --beta 0.0 \
    --vit-model ViT-B-16 --ckpt-dir /path/to/checkpoints/ViT-B-16 \
    --eval-dir /path/to/task_vectors
```

### RoBERTa experiments

```bash
# Plain baseline (Task Arithmetic)
python run_roberta.py --method ta --mode plain --model-size base \
    --ckpt-dir /path/to/roberta_checkpoints

# CUSD + Task Arithmetic (beta=0)
python run_roberta.py --method ta --mode cusd --beta 0.0 \
    --model-size base --ckpt-dir /path/to/roberta_checkpoints
```

### LLM experiments

```bash
# Plain baseline (Task Arithmetic)
python run_llm.py --method ta --mode plain --model 3b \
    --base-path /path/to/Llama-3.2-3B \
    --tv-dir /path/to/task_vectors

# CUSD + Task Arithmetic (beta=0.3)
python run_llm.py --method ta --mode cusd --beta 0.3 --model 3b \
    --base-path /path/to/Llama-3.2-3B \
    --tv-dir /path/to/task_vectors

```

## Acknowledgments

We use checkpoints and evaluation code from:

- [Task Vectors](https://github.com/mlfoundations/task_vectors)
- [WUDI-Merging](https://github.com/nathanielyvo/WUDI-Merging)
- [MergeBench](https://huggingface.co/MergeBench)

We thank the authors for making their resources publicly available.
