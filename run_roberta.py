#!/usr/bin/env python3
"""
run_roberta.py -- RoBERTa GLUE 8-task merging + evaluation for CUSD.

"""

import os
import sys
import json
import time
import re
import gc
import argparse
import traceback
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import partial
from copy import deepcopy

import torch

sys.path.insert(0, str(Path(__file__).parent))

from cusd.framework import cusd_merge_layer
from methods import METHODS

# ==============================================================================
# Constants
# ==============================================================================
GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]

GLUE_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "stsb": 1,
    "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2,
}

GLUE_METRICS_MAP = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "averaged_scores",
    "stsb": "averaged_scores",
    "qqp": "averaged_scores",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
}

GLUE_DATA_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

# Parameter name patterns to exclude from merging
SKIP_PATTERNS = [
    r".*classifier.*",
    r".*embeddings\.word_embeddings.*",
    r".*embeddings\.position_embeddings.*",
    r".*embeddings\.token_type_embeddings.*",
]


# ==============================================================================
# Key scope filtering
# ==============================================================================
def should_skip_key(key):
    """Check if a parameter key should be excluded from merging."""
    return any(re.match(pat, key) for pat in SKIP_PATTERNS)


def compute_task_vectors(base_state, finetuned_states):
    """
    Compute task vectors: tv[key] = finetuned[key] - base[key].
    Only keeps 2D keys that are not in SKIP_PATTERNS.
    """
    all_task_names = sorted(finetuned_states.keys())
    task_vectors = {}

    for task_name in all_task_names:
        ft_state = finetuned_states[task_name]
        tv = {}
        for key in ft_state:
            if key not in base_state:
                continue
            if should_skip_key(key):
                continue
            delta = ft_state[key].float() - base_state[key].float()
            if delta.dim() == 2:
                tv[key] = delta
        task_vectors[task_name] = tv
        print(f"  {task_name}: {len(tv)} 2D merge keys", flush=True)

    if not task_vectors:
        return {}, []

    common_keys = set(task_vectors[all_task_names[0]].keys())
    for tn in all_task_names[1:]:
        common_keys &= set(task_vectors[tn].keys())
    merge_keys = sorted(common_keys)

    for tn in all_task_names:
        task_vectors[tn] = {k: task_vectors[tn][k] for k in merge_keys}

    print(f"  Common 2D merge keys: {len(merge_keys)}", flush=True)
    return task_vectors, merge_keys


# ==============================================================================
# GLUE Evaluation
# ==============================================================================
def compute_single_metrics(eval_pred, dataset_name):
    """Compute metrics for a single GLUE task."""
    import evaluate

    if eval_pred.predictions.shape[1] > 1:
        predictions = np.argmax(eval_pred.predictions, axis=1)
    else:
        predictions = eval_pred.predictions.squeeze(axis=1)

    metric_func = evaluate.load("glue", dataset_name)
    result = metric_func.compute(
        predictions=predictions,
        references=eval_pred.label_ids,
    )
    if len(result) > 1:
        result["averaged_scores"] = np.mean(list(result.values())).item()
    else:
        result["averaged_scores"] = list(result.values())[0]
    return result


def load_glue_dataset(tokenizer, dataset_name, split="validation"):
    """Load and tokenize a GLUE dataset split."""
    import datasets

    if split != "train":
        split = "validation_matched" if dataset_name == "mnli" else "validation"

    dataset = datasets.load_dataset("glue", dataset_name, split=split)

    sentence1_key, sentence2_key = GLUE_DATA_KEYS[dataset_name]
    dataset = dataset.map(
        lambda examples: tokenizer(
            text=examples[sentence1_key],
            text_pair=examples[sentence2_key] if sentence2_key else None,
            max_length=128,
            truncation=True,
        ),
        batched=True,
    )
    return dataset


def evaluate_glue(model, tokenizer, merged_encoder_state, classifier_heads,
                  task_names):
    """
    Evaluate merged encoder + task-specific classifier heads on GLUE tasks.

    For each task:
      1. Load merged encoder weights
      2. Attach the task-specific classifier head
      3. Evaluate on GLUE validation set
    """
    from transformers import Trainer, TrainingArguments

    per_task_results = {}

    for task_name in task_names:
        print(f"\n--- Evaluating {task_name} ---", flush=True)
        task_start = time.time()

        try:
            num_labels = GLUE_NUM_LABELS[task_name]
            model.config.num_labels = num_labels
            model.num_labels = num_labels
            model.config.problem_type = (
                "regression" if task_name == "stsb" else "single_label_classification"
            )

            current_state = model.state_dict()
            for key, val in merged_encoder_state.items():
                if key in current_state:
                    current_state[key] = val.to(current_state[key].dtype).to(
                        current_state[key].device)
            model.load_state_dict(current_state, strict=False)

            classifier_head = classifier_heads[task_name]
            if isinstance(classifier_head, dict):
                model.classifier.load_state_dict(classifier_head)
            else:
                model.classifier = classifier_head.to(model.device)

            test_dataset = load_glue_dataset(tokenizer, task_name, split="validation")

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir="/tmp/glue_eval",
                    per_device_eval_batch_size=128,
                    report_to=[],
                    dataloader_num_workers=0,
                ),
                eval_dataset=test_dataset,
                compute_metrics=partial(compute_single_metrics, dataset_name=task_name),
                processing_class=tokenizer,
            )

            metrics = trainer.evaluate()
            task_elapsed = time.time() - task_start

            primary_key = f"eval_{GLUE_METRICS_MAP[task_name]}"
            primary_score = metrics.get(primary_key)
            if primary_score is not None:
                primary_score = primary_score * 100

            per_task_results[task_name] = {
                "primary_metric": GLUE_METRICS_MAP[task_name],
                "primary_score": primary_score,
                "elapsed_sec": round(task_elapsed, 1),
            }
            print(f"  {task_name}: {primary_score:.2f} ({task_elapsed:.1f}s)", flush=True)

        except Exception as e:
            task_elapsed = time.time() - task_start
            print(f"  {task_name}: FAILED - {e}", flush=True)
            traceback.print_exc()
            per_task_results[task_name] = {
                "primary_metric": GLUE_METRICS_MAP[task_name],
                "primary_score": None,
                "elapsed_sec": round(task_elapsed, 1),
                "error": str(e),
            }

    return per_task_results


# ==============================================================================
# Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CUSD RoBERTa GLUE Merging")
    parser.add_argument("--model-size", choices=["base", "large"], required=True)
    parser.add_argument("--method", choices=["ta", "dare", "ties", "consensus"],
                        required=True)
    parser.add_argument("--mode", choices=["cusd", "plain"], required=True)

    # CUSD parameters
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Unique residual scaling (CUSD mode)")
    parser.add_argument("--pi", type=float, default=1e-3,
                        help="HO-GSVD regularization")
    parser.add_argument("--classify-mode", default="abs_std",
                        choices=["rel_std", "abs_std"],
                        help="Dispersion scoring strategy")
    parser.add_argument("--breakpoint-method", default="kde",
                        choices=["kde", "gmm", "fixed_value"],
                        help="Breakpoint detection method")
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--fallback-quantile", type=float, default=0.9)

    # TA scaling
    parser.add_argument("--scaling", type=float, default=0.4,
                        help="TA scaling factor")

    # Method-specific parameters
    parser.add_argument("--drop-rate", type=float, default=0.5)
    parser.add_argument("--trim-ratio", type=float, default=0.3)
    parser.add_argument("--consensus-k", type=int, default=2)
    parser.add_argument("--consensus-lamda", type=float, default=0.2)

    # Paths
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="RoBERTa checkpoint directory (contains task subdirs)")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def make_task_id(args):
    beta_str = f"_b{args.beta}" if args.mode == "cusd" else ""
    method_params = ""
    if args.method == "dare":
        method_params = f"_p{args.drop_rate}"
    elif args.method == "ties":
        method_params = f"_k{args.trim_ratio}"
    elif args.method == "consensus":
        method_params = f"_k{args.consensus_k}_l{args.consensus_lamda}"

    scaling_str = f"_s{args.scaling}" if args.scaling != 1.0 else ""
    return (f"roberta_{args.mode}_{args.method}{method_params}"
            f"{scaling_str}{beta_str}_{args.model_size}")


def main():
    args = parse_args()
    task_id = make_task_id(args)
    overall_start = time.time()

    ckpt_dir = Path(args.ckpt_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = f"roberta-{args.model_size}"
    ckpt_subdir = f"{base_model_name}_lr1e-05"

    print(f"Device: {device}", flush=True)
    print(f"Config: method={args.method}, mode={args.mode}, model={base_model_name}, "
          f"scaling={args.scaling}, beta={args.beta}", flush=True)

    # Load base model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=3, torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_state = {k: v.float() for k, v in base_model.state_dict().items()}

    # Load finetuned checkpoints and compute task vectors
    print(f"\nLoading finetuned checkpoints from {ckpt_dir}", flush=True)
    finetuned_states = {}
    classifier_heads = {}

    for task_name in GLUE_TASKS:
        ft_path = ckpt_dir / task_name / ckpt_subdir
        if not ft_path.exists():
            print(f"  WARNING: {ft_path} not found, skipping {task_name}", flush=True)
            continue

        ft_model = AutoModelForSequenceClassification.from_pretrained(
            str(ft_path), torch_dtype=torch.float32,
        )
        ft_state = {k: v.float() for k, v in ft_model.state_dict().items()}
        finetuned_states[task_name] = ft_state
        classifier_heads[task_name] = deepcopy(ft_model.classifier)
        del ft_model
        gc.collect()

    active_tasks = sorted(finetuned_states.keys())
    K = len(active_tasks)
    print(f"  Loaded {K} finetuned models", flush=True)

    if K == 0:
        raise RuntimeError("No finetuned checkpoints found.")

    task_vectors_dict, merge_keys = compute_task_vectors(base_state, finetuned_states)
    del finetuned_states
    gc.collect()

    # Build method kwargs
    method_fn = METHODS[args.method]
    method_kwargs = {}
    if args.method == "ta":
        method_kwargs = {"scaling": args.scaling}
    elif args.method == "dare":
        method_kwargs = {"drop_rate": args.drop_rate, "scaling": args.scaling,
                         "seed": args.seed}
    elif args.method == "ties":
        method_kwargs = {"trim_ratio": args.trim_ratio, "scaling": args.scaling}
    elif args.method == "consensus":
        method_kwargs = {"scaling": args.scaling, "k": args.consensus_k,
                         "lamda": args.consensus_lamda}

    # Merge
    merged_tv = {}
    decomp_stats = {}
    total_common = 0
    total_unique = 0
    total_dims = 0

    for ki, key in enumerate(merge_keys):
        tvs = [task_vectors_dict[ds][key] for ds in active_tasks]

        if args.mode == "cusd":
            merged, stats = cusd_merge_layer(
                tvs, method_fn, method_kwargs,
                pi=args.pi, breakpoint_method=args.breakpoint_method,
                beta=args.beta, device=device,
                classify_mode=args.classify_mode,
                fallback_quantile=args.fallback_quantile,
                fixed_threshold=args.fixed_threshold,
            )
            merged_tv[key] = merged
            total_common += stats.get("n_common", 0)
            total_unique += stats.get("n_unique", 0)
            total_dims += stats.get("n", 0)
            decomp_stats[key] = {k: v for k, v in stats.items()
                                 if isinstance(v, (int, float, str, bool, type(None)))}
        else:
            merged_tv[key] = method_fn(tvs, **method_kwargs)

        if (ki + 1) % 20 == 0 or ki == len(merge_keys) - 1:
            print(f"  Key {ki+1}/{len(merge_keys)}", flush=True)

    del task_vectors_dict
    gc.collect()

    # Build merged encoder state (exclude classifier keys)
    merged_encoder_state = {
        k: v.clone() for k, v in base_state.items()
        if not re.match(r".*classifier.*", k)
    }
    for key, tv in merged_tv.items():
        if key in merged_encoder_state:
            merged_encoder_state[key] = merged_encoder_state[key] + tv

    del merged_tv
    gc.collect()

    # Evaluate
    base_model = base_model.to(device)
    torch.cuda.empty_cache()

    per_task_results = evaluate_glue(
        base_model, tokenizer, merged_encoder_state, classifier_heads, active_tasks,
    )

    del base_model, tokenizer, merged_encoder_state, classifier_heads
    gc.collect()
    torch.cuda.empty_cache()

    # Compute summary
    overall_elapsed = time.time() - overall_start

    per_task_scores = {}
    for tn, tr in per_task_results.items():
        if tr["primary_score"] is not None:
            per_task_scores[tn] = tr["primary_score"]

    avg_score = float(np.mean(list(per_task_scores.values()))) if per_task_scores else 0.0

    cusd_summary = {}
    if args.mode == "cusd" and total_dims > 0:
        cusd_summary = {
            "total_common": total_common,
            "total_unique": total_unique,
            "total_dims": total_dims,
            "common_frac": round(total_common / total_dims, 4),
            "unique_frac": round(total_unique / total_dims, 4),
        }

    # Save results
    full_results = {
        "task_id": task_id,
        "method": args.method,
        "mode": args.mode,
        "model_size": args.model_size,
        "base_model": base_model_name,
        "beta": args.beta if args.mode == "cusd" else None,
        "n_tasks": K,
        "active_tasks": active_tasks,
        "args": vars(args),
        "per_task_results": per_task_results,
        "per_task_scores": per_task_scores,
        "avg_score": round(avg_score, 2),
        "decomposition": cusd_summary,
        "total_elapsed_sec": round(overall_elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = results_dir / f"{task_id}.json"
    out_path.write_text(json.dumps(full_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {task_id}", flush=True)
    print(f"{'='*60}", flush=True)
    for tn in active_tasks:
        score = per_task_scores.get(tn, "FAIL")
        score_str = f"{score:.2f}" if isinstance(score, float) else score
        print(f"  {tn:8s}: {score_str}", flush=True)
    print(f"  {'Average':8s}: {avg_score:.2f}", flush=True)
    if cusd_summary:
        print(f"  CUSD: {cusd_summary.get('common_frac', 0):.1%} common, "
              f"{cusd_summary.get('unique_frac', 0):.1%} unique", flush=True)
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
