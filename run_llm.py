#!/usr/bin/env python3
"""
run_llm.py -- LLM merging + evaluation for CUSD.

"""

import sys
import os
import json
import time
import gc
import argparse
import traceback
from pathlib import Path
from datetime import datetime

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cusd.framework import cusd_merge_full, plain_merge_full
from methods import METHODS


# ==============================================================================
# Configuration
# ==============================================================================
EXPERT_CONFIGS = {
    3: ["math", "multilingual", "coding"],
    5: ["instruction", "math", "multilingual", "coding", "safety"],
}

EVAL_DIMENSIONS_3 = {
    "math": {
        "tasks": ["gsm8k"],
        "primary_metric": "exact_match,flexible-extract",
    },
    "multilingual": {
        "tasks": ["mmlu"],
        "primary_metric": "acc,none",
    },
    "coding": {
        "tasks": ["humaneval"],
        "primary_metric": "pass@1,none",
    },
}

EVAL_DIMENSIONS_5 = {
    "instruction": {"tasks": ["ifeval"], "primary_metric": "prompt_level_strict_acc,none"},
    "math_gsm8k": {"tasks": ["gsm8k"], "primary_metric": "exact_match,flexible-extract"},
    "math_math": {"tasks": ["hendrycks_math"], "primary_metric": "exact_match,flexible-extract"},
    "multilingual_mmlu": {"tasks": ["mmlu"], "primary_metric": "acc,none"},
    "multilingual_arc": {"tasks": ["arc_challenge"], "primary_metric": "acc_norm,none"},
    "coding_humaneval": {"tasks": ["humaneval"], "primary_metric": "pass@1,none"},
    "coding_mbpp": {"tasks": ["mbpp"], "primary_metric": "pass@1,none"},
    "safety_truthfulqa": {"tasks": ["truthfulqa_mc2"], "primary_metric": "acc,none"},
}

SKIP_KEYS = ["embed_tokens", "lm_head"]
SEED = 42


# ==============================================================================
# Task vector loading
# ==============================================================================
def load_task_vectors(tv_dir, expert_names):
    """Load task vectors for specified experts (2D keys only)."""
    print(f"Loading {len(expert_names)} task vectors from {tv_dir}", flush=True)

    raw_tvs = []
    all_keys_list = []
    for name in expert_names:
        tv_path = Path(tv_dir) / f"{name}.pt"
        if not tv_path.exists():
            raise FileNotFoundError(f"Task vector not found: {tv_path}")
        tv = torch.load(str(tv_path), map_location="cpu", weights_only=True)
        raw_tvs.append(tv)
        all_keys_list.append(set(tv.keys()))
        print(f"  {name}: {len(tv)} keys", flush=True)

    common_keys = all_keys_list[0]
    for ks in all_keys_list[1:]:
        common_keys = common_keys & ks

    filtered_keys = sorted([
        k for k in common_keys
        if not any(skip in k for skip in SKIP_KEYS)
    ])

    task_vectors = []
    for tv in raw_tvs:
        task_vectors.append({k: tv[k].float() for k in filtered_keys})
    del raw_tvs
    gc.collect()

    # Restricted scope: only 2D keys
    keys_to_remove = [k for k in filtered_keys if task_vectors[0][k].dim() != 2]
    if keys_to_remove:
        print(f"  Filtering out {len(keys_to_remove)} non-2D keys", flush=True)
        for tvd in task_vectors:
            for k in keys_to_remove:
                del tvd[k]
        filtered_keys = [k for k in filtered_keys if k not in set(keys_to_remove)]

    print(f"  Final: {len(filtered_keys)} 2D keys", flush=True)
    return task_vectors, filtered_keys


# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_merged_tv(merged_tv, base_model_path, eval_dims, config_name,
                       device="cuda", limit=None, seed=42):
    """Apply merged TV to base model and evaluate with lm-eval."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {config_name}", flush=True)
    if limit:
        print(f"  PILOT MODE: limit={limit}", flush=True)
    print(f"{'='*60}", flush=True)

    eval_start = time.time()

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_state = base_model.state_dict()

    applied_count = 0
    for key in base_state:
        if key in merged_tv:
            base_state[key] = base_state[key] + merged_tv[key].to(base_state[key].dtype)
            applied_count += 1
    print(f"  Applied {applied_count} TV keys to base model", flush=True)

    base_model.load_state_dict(base_state)
    del base_state
    gc.collect()

    base_model = base_model.to(device)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_wrapper = HFLM(pretrained=base_model, tokenizer=tokenizer, batch_size=32)

    results = {}
    dimension_scores = {}

    for dim_name, dim_config in eval_dims.items():
        dim_start = time.time()
        tasks = dim_config["tasks"]
        primary_metric = dim_config["primary_metric"]

        print(f"\n--- {dim_name} ({tasks}) ---", flush=True)

        try:
            eval_kwargs = {
                "model": model_wrapper,
                "tasks": tasks,
                "num_fewshot": 0,
                "random_seed": seed,
                "numpy_random_seed": seed,
                "torch_random_seed": seed,
                "confirm_run_unsafe_code": True,
            }
            if limit:
                eval_kwargs["limit"] = limit

            eval_results = lm_eval.simple_evaluate(**eval_kwargs)
            task_results = eval_results.get("results", {})
            dim_elapsed = time.time() - dim_start

            primary_score = _extract_score(task_results, tasks, primary_metric, dim_name)

            results[dim_name] = {
                "tasks": task_results,
                "primary_score": primary_score,
                "elapsed_sec": round(dim_elapsed, 1),
                "status": "success",
            }
            if primary_score is not None:
                dimension_scores[dim_name] = primary_score
            print(f"  {dim_name}: {primary_score} ({dim_elapsed:.1f}s)", flush=True)

        except Exception as e:
            print(f"  {dim_name}: FAILED - {e}", flush=True)
            traceback.print_exc()
            results[dim_name] = {"primary_score": None, "status": "failed", "error": str(e)}

    del base_model, model_wrapper, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    total_elapsed = time.time() - eval_start
    valid_scores = {k: v for k, v in dimension_scores.items() if v is not None}
    avg_score = float(np.mean(list(valid_scores.values()))) if valid_scores else 0.0

    summary = {
        "dimension_scores": dimension_scores,
        "avg_score": avg_score,
        "valid_dimensions": len(valid_scores),
        "total_elapsed_sec": round(total_elapsed, 1),
    }
    return results, summary


def _extract_score(task_results, tasks, primary_metric, dim_name):
    """Extract primary score from lm_eval results."""
    for task_name, task_data in task_results.items():
        if task_name.startswith("mmlu_") and task_name != "mmlu":
            continue
        if primary_metric in task_data:
            return task_data[primary_metric]
        if "coding" in dim_name:
            for mk in ["pass@1,create_test", "pass_at_1,none", "pass@1,none", "pass@1"]:
                if mk in task_data:
                    return task_data[mk]

    main_task = tasks[0]
    if main_task in task_results:
        for mk in ["acc,none", "pass@1,none", "exact_match,flexible-extract",
                    "prompt_level_strict_acc,none"]:
            if mk in task_results[main_task]:
                return task_results[main_task][mk]
    return None


# ==============================================================================
# Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CUSD LLM Merging")
    parser.add_argument("--method", choices=["ta", "dare", "ties", "consensus"],
                        required=True)
    parser.add_argument("--mode", choices=["cusd", "plain"], required=True,
                        help="cusd = CUSD decomposition, plain = baseline")
    parser.add_argument("--model", default="3b",
                        help="Model identifier (for output naming)")
    parser.add_argument("--experts", type=int, choices=[3, 5], default=3)

    # Paths (required)
    parser.add_argument("--base-path", type=str, required=True,
                        help="Path to base model (HuggingFace format)")
    parser.add_argument("--tv-dir", type=str, required=True,
                        help="Directory containing task vector .pt files")

    # CUSD parameters
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Unique residual scaling (CUSD mode)")
    parser.add_argument("--pi", type=float, default=1e-3,
                        help="HO-GSVD regularization")
    parser.add_argument("--classify-mode", default="rel_std",
                        choices=["rel_std", "abs_std"],
                        help="Dispersion scoring strategy")
    parser.add_argument("--breakpoint-method", default="kde",
                        choices=["kde", "gmm", "fixed_value"],
                        help="Breakpoint detection method")
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Fixed threshold (breakpoint-method=fixed_value)")
    parser.add_argument("--fallback-quantile", type=float, default=0.9,
                        help="Fallback quantile when no KDE valley found")

    # Method-specific parameters
    parser.add_argument("--ta-scaling", type=float, default=0.4,
                        help="TA scaling factor")
    parser.add_argument("--drop-rate", type=float, default=0.5,
                        help="DARE drop rate")
    parser.add_argument("--trim-ratio", type=float, default=0.2,
                        help="TIES trim ratio")
    parser.add_argument("--consensus-k", type=int, default=2,
                        help="Consensus TA: min experts that must agree")
    parser.add_argument("--consensus-lamda", type=float, default=0.2,
                        help="Consensus TA: per-expert tall mask threshold")

    # Evaluation
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluation sample limit (None = full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Output directory")

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

    scaling_str = f"_s{args.ta_scaling}" if args.ta_scaling != 1.0 else ""
    return f"{args.mode}_{args.method}{method_params}{scaling_str}{beta_str}_{args.model}_{args.experts}exp"


def main():
    args = parse_args()
    overall_start = time.time()
    task_id = make_task_id(args)

    expert_names = EXPERT_CONFIGS[args.experts]
    eval_dims = EVAL_DIMENSIONS_3 if args.experts == 3 else EVAL_DIMENSIONS_5

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Config: method={args.method}, mode={args.mode}, model={args.model}, "
          f"experts={args.experts}, beta={args.beta}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load task vectors
    task_vectors, filtered_keys = load_task_vectors(args.tv_dir, expert_names)

    # Build method kwargs
    method_fn = METHODS[args.method]
    method_kwargs = {}
    if args.method == "ta":
        method_kwargs = {"scaling": args.ta_scaling}
    elif args.method == "dare":
        method_kwargs = {"drop_rate": args.drop_rate, "scaling": args.ta_scaling, "seed": args.seed}
    elif args.method == "ties":
        method_kwargs = {"trim_ratio": args.trim_ratio, "scaling": args.ta_scaling}
    elif args.method == "consensus":
        method_kwargs = {"scaling": args.ta_scaling, "k": args.consensus_k,
                         "lamda": args.consensus_lamda}

    # Merge
    if args.mode == "cusd":
        print(f"\nCUSD merge: beta={args.beta}, classify={args.classify_mode}", flush=True)
        merged_tv, merge_stats = cusd_merge_full(
            task_vectors, method_fn, method_kwargs,
            pi=args.pi, beta=args.beta,
            breakpoint_method=args.breakpoint_method,
            device=device,
            classify_mode=args.classify_mode,
            fallback_quantile=args.fallback_quantile,
            fixed_threshold=args.fixed_threshold,
        )
    else:
        print(f"\nPlain merge: {args.method.upper()}", flush=True)
        merged_tv, merge_stats = plain_merge_full(
            task_vectors, method_fn, method_kwargs,
        )

    del task_vectors
    gc.collect()

    # Evaluate
    print("\nStarting evaluation...", flush=True)
    eval_results, eval_summary = evaluate_merged_tv(
        merged_tv, args.base_path, eval_dims,
        config_name=task_id, device=device, limit=args.limit, seed=args.seed,
    )

    del merged_tv
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    overall_elapsed = time.time() - overall_start
    full_results = {
        "task_id": task_id,
        "method": args.method,
        "mode": args.mode,
        "model": args.model,
        "experts": args.experts,
        "args": vars(args),
        "merge_stats": merge_stats,
        "eval_results": eval_results,
        "eval_summary": eval_summary,
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
    if args.mode == "cusd":
        print(f"Decomposition: {merge_stats.get('common_fraction', 0):.1%} common, "
              f"{merge_stats.get('unique_fraction', 0):.1%} unique", flush=True)
    for dim, score in eval_summary.get("dimension_scores", {}).items():
        print(f"  {dim:15s}: {score}", flush=True)
    print(f"  {'Average':15s}: {eval_summary.get('avg_score', 0):.4f}", flush=True)
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
