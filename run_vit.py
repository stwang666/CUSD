#!/usr/bin/env python3
"""
run_vit.py -- ViT merging + evaluation for CUSD.
"""

import os
import sys
import json
import time
import gc
import argparse
import pickle
import traceback
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch

sys.path.insert(0, str(Path(__file__).parent))

from cusd.framework import cusd_merge_layer
from methods import METHODS

TASK_NAMES = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]

# Only merge these weight patterns (attention + MLP projections)
MERGE_KEY_PATTERNS = [
    "attn.in_proj_weight",
    "attn.out_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]


def is_merge_key(key):
    return any(pat in key for pat in MERGE_KEY_PATTERNS)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except RuntimeError:
        with open(path, 'rb') as f:
            return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="CUSD ViT Merging")
    parser.add_argument("--method", choices=["ta", "dare", "ties", "consensus"],
                        required=True)
    parser.add_argument("--mode", choices=["cusd", "plain"], required=True)
    parser.add_argument("--vit-model", default="ViT-B-16")

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

    # TA scaling (lambda)
    parser.add_argument("--scaling", type=float, default=0.4,
                        help="TA scaling factor")

    # Method-specific parameters
    parser.add_argument("--drop-rate", type=float, default=0.5)
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--consensus-k", type=int, default=2)
    parser.add_argument("--consensus-lamda", type=float, default=0.2)

    # Paths
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Checkpoint directory (contains zeroshot.pt and task subdirs)")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Directory containing ViT eval code (args.py, eval.py). "
                             "If not set, must have eval code on PYTHONPATH.")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)

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

    scaling_str = f"_l{args.scaling}" if args.scaling != 0.4 else ""
    return f"vit_{args.mode}_{args.method}{method_params}{scaling_str}{beta_str}_{args.vit_model.replace('-', '')}"


def main():
    args = parse_args()
    task_id = make_task_id(args)
    overall_start = time.time()

    ckpt_dir = Path(args.ckpt_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Add eval code to path
    if args.eval_dir:
        sys.path.insert(0, args.eval_dir)

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Config: method={args.method}, mode={args.mode}, "
          f"model={args.vit_model}, beta={args.beta}, scaling={args.scaling}", flush=True)

    # Load pretrained model
    pretrained_path = ckpt_dir / "zeroshot.pt"
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    pretrained_model = load_checkpoint(str(pretrained_path))
    pretrained_sd = pretrained_model.state_dict()

    # Load finetuned models and compute task vectors
    task_vectors_dict = {}
    for ds in TASK_NAMES:
        ft_path = ckpt_dir / ds / "finetuned.pt"
        if not ft_path.exists():
            print(f"  WARNING: {ft_path} not found, skipping {ds}", flush=True)
            continue
        ft_model = load_checkpoint(str(ft_path))
        ft_sd = ft_model.state_dict()
        tv = {}
        for key in ft_sd:
            if key not in pretrained_sd:
                continue
            if ft_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            if not is_merge_key(key):
                continue
            tv[key] = (ft_sd[key] - pretrained_sd[key]).float()
        task_vectors_dict[ds] = tv
        del ft_model, ft_sd
        print(f"  {ds}: {len(tv)} merge keys", flush=True)

    active_tasks = sorted(task_vectors_dict.keys())
    K = len(active_tasks)
    if K == 0:
        raise RuntimeError("No finetuned checkpoints found.")
    merge_keys = sorted(task_vectors_dict[active_tasks[0]].keys())
    print(f"  {K} tasks, {len(merge_keys)} merge keys", flush=True)

    # Build method kwargs
    method_fn = METHODS[args.method]
    method_kwargs = {}
    if args.method == "ta":
        method_kwargs = {"scaling": args.scaling}
    elif args.method == "dare":
        method_kwargs = {"drop_rate": args.drop_rate, "scaling": args.scaling, "seed": args.seed}
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

        if (ki + 1) % max(12, len(merge_keys) // 10) == 0 or ki == len(merge_keys) - 1:
            print(f"  Key {ki+1}/{len(merge_keys)}", flush=True)

    del task_vectors_dict
    gc.collect()

    # Apply merged TV to pretrained model
    merged_sd = deepcopy(pretrained_sd)
    for key, tv in merged_tv.items():
        if key in merged_sd:
            merged_sd[key] = merged_sd[key] + tv.to(merged_sd[key].dtype)

    merged_model = deepcopy(pretrained_model)
    merged_model.load_state_dict(merged_sd)
    merged_model = merged_model.cuda()
    del merged_sd, merged_tv
    gc.collect()

    # Evaluate using eval code
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    if eval_dir:
        os.chdir(str(eval_dir))

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from args import parse_arguments
    eval_args = parse_arguments()
    sys.argv = saved_argv
    eval_args.model = args.vit_model
    eval_args.data_location = 'data'
    eval_args.save = str(ckpt_dir)
    eval_args.batch_size = 128

    from eval import eval_single_dataset_with_loss

    # Finetuned expert accuracies (for retention rate)
    print("\n--- Finetuned expert accuracies ---", flush=True)
    finetuned_accs = {}
    for ds in active_tasks:
        ft_path = ckpt_dir / ds / "finetuned.pt"
        expert_model = load_checkpoint(str(ft_path))
        metrics, _ = eval_single_dataset_with_loss(expert_model, ds, eval_args)
        finetuned_accs[ds] = metrics.get('top1', 0.0) * 100
        print(f"  {ds}: {finetuned_accs[ds]:.2f}%", flush=True)
        del expert_model

    # Merged model accuracies
    print("\n--- Merged model accuracies ---", flush=True)
    merged_accs = {}
    for ds in active_tasks:
        metrics, _ = eval_single_dataset_with_loss(merged_model, ds, eval_args)
        merged_accs[ds] = metrics.get('top1', 0.0) * 100
        print(f"  {ds}: {merged_accs[ds]:.2f}%", flush=True)

    avg_finetuned = np.mean(list(finetuned_accs.values()))
    avg_merged = np.mean(list(merged_accs.values()))
    retention = avg_merged / avg_finetuned * 100 if avg_finetuned > 0 else 0.0

    # Save results
    overall_elapsed = time.time() - overall_start

    per_task = []
    for ds in active_tasks:
        ft = finetuned_accs.get(ds, 0)
        mg = merged_accs.get(ds, 0)
        ret = mg / ft * 100 if ft > 0 else 0
        per_task.append({
            "task": ds, "finetuned_acc": round(ft, 2),
            "merged_acc": round(mg, 2), "retention_pct": round(ret, 2),
        })

    cusd_summary = {}
    if args.mode == "cusd" and total_dims > 0:
        cusd_summary = {
            "total_common": total_common,
            "total_unique": total_unique,
            "total_dims": total_dims,
            "common_frac": round(total_common / total_dims, 4),
            "unique_frac": round(total_unique / total_dims, 4),
        }

    full_results = {
        "task_id": task_id,
        "method": args.method,
        "mode": args.mode,
        "vit_model": args.vit_model,
        "beta": args.beta if args.mode == "cusd" else None,
        "args": vars(args),
        "per_task": per_task,
        "avg_finetuned_acc": round(avg_finetuned, 2),
        "avg_merged_acc": round(avg_merged, 2),
        "retention_pct": round(retention, 2),
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
    for r in per_task:
        print(f"  {r['task']:12s}  ft={r['finetuned_acc']:.1f}%  "
              f"merged={r['merged_acc']:.1f}%  ret={r['retention_pct']:.1f}%", flush=True)
    print(f"  {'Avg':12s}  ft={avg_finetuned:.1f}%  merged={avg_merged:.1f}%  "
          f"retention={retention:.1f}%", flush=True)
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
