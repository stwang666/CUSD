"""
CUSD merge pipeline.
"""

import time
import gc
from typing import List, Dict, Optional, Tuple, Callable

import torch
import numpy as np

from .hogsvd import hogsvd, classify_subspaces, decompose_task_vectors, CLASSIFY_MODES


def cusd_merge_layer(
    task_vectors_layer: List[torch.Tensor],
    method_fn: Callable,
    method_kwargs: dict,
    pi: float = 1e-3,
    breakpoint_method: str = "kde",
    beta: float = 0.0,
    device: str = "cuda",
    classify_mode: str = "abs_std",
    fallback_quantile: float = 0.9,
    fixed_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    CUSD merge for a single 2D layer.

    Args:
        task_vectors_layer: list of K tensors (m, n).
        method_fn: merge function with signature merge(tvs, **kwargs) -> Tensor.
        method_kwargs: kwargs passed to method_fn.
        pi: HO-GSVD regularization.
        breakpoint_method: "kde", "gmm", or "fixed_value".
        beta: unique residual scaling (0 = discard unique).
        device: compute device.
        classify_mode: dispersion scoring strategy.
        fallback_quantile: quantile for non-bimodal fallback.
        fixed_threshold: threshold for breakpoint_method="fixed_value".

    Returns:
        merged: merged tensor for this layer.
        stats: dict with decomposition statistics.
    """
    if task_vectors_layer[0].dim() < 2:
        merged = method_fn(task_vectors_layer, **method_kwargs)
        return merged, {"type": "1d", "method": "direct"}

    if task_vectors_layer[0].dim() > 2:
        merged = method_fn(task_vectors_layer, **method_kwargs)
        return merged, {"type": f"{task_vectors_layer[0].dim()}d", "method": "direct"}

    n = task_vectors_layer[0].shape[1]

    # Step 1: HO-GSVD
    V, eigenvalues, Sigma_list, hogsvd_diag = hogsvd(task_vectors_layer, pi=pi, device=device)

    # Step 2: Classify directions
    common_mask, unique_mask, theta, cls_diag = classify_subspaces(
        Sigma_list,
        breakpoint_method=breakpoint_method,
        classify_mode=classify_mode,
        fallback_quantile=fallback_quantile,
        fixed_threshold=fixed_threshold,
    )

    n_common = int(common_mask.sum())
    n_unique = int(unique_mask.sum())

    stats = {
        "type": "2d",
        "n": n,
        "n_common": n_common,
        "n_unique": n_unique,
        "common_frac": round(n_common / n, 4) if n > 0 else 0,
        "theta": round(float(theta), 6),
        "classify_mode": classify_mode,
        "is_bimodal": cls_diag["breakpoint_info"].get("is_bimodal", False),
        "non_bimodal_all_common": cls_diag.get("non_bimodal_all_common", False),
        "asym_norm": hogsvd_diag.get("asym_norm"),
    }

    # Step 3: Decompose
    C_list, R_list = decompose_task_vectors(task_vectors_layer, V, common_mask)

    # Step 4: Merge common parts
    merged_common = method_fn(C_list, **method_kwargs)

    # Step 5: Unique residual contribution
    if beta != 0.0 and n_unique > 0:
        unique_sum = torch.zeros_like(R_list[0])
        for R_i in R_list:
            unique_sum += R_i
        unique_contribution = beta * unique_sum
    else:
        unique_contribution = torch.zeros_like(merged_common)

    # Step 6: Reconstruct
    merged = merged_common + unique_contribution

    del V, eigenvalues, Sigma_list, C_list, R_list
    if device == "cuda":
        torch.cuda.empty_cache()

    return merged, stats


def cusd_merge_full(
    task_vectors: List[Dict[str, torch.Tensor]],
    method_fn: Callable,
    method_kwargs: dict,
    pi: float = 1e-3,
    breakpoint_method: str = "kde",
    beta: float = 0.0,
    device: str = "cuda",
    skip_keys: Optional[List[str]] = None,
    classify_mode: str = "abs_std",
    fallback_quantile: float = 0.9,
    fixed_threshold: Optional[float] = None,
    log_interval: int = 20,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    CUSD merge across all layers.

    Args:
        task_vectors: list of K dicts {key: tensor}.
        method_fn: merge function.
        method_kwargs: kwargs for method.
        pi: HO-GSVD regularization.
        breakpoint_method: "kde", "gmm", or "fixed_value".
        beta: unique residual scaling.
        device: compute device.
        skip_keys: key patterns to skip entirely (not merged).
        classify_mode: dispersion scoring strategy.
        fallback_quantile: quantile for non-bimodal fallback.
        fixed_threshold: threshold for breakpoint_method="fixed_value".
        log_interval: print progress every N layers.

    Returns:
        merged_tv: dict {key: tensor}.
        merge_stats: dict with overall statistics.
    """
    if skip_keys is None:
        skip_keys = ["embed_tokens", "lm_head"]

    K = len(task_vectors)
    all_keys = sorted(task_vectors[0].keys())

    filtered_keys = [
        k for k in all_keys
        if not any(skip in k for skip in skip_keys)
    ]

    keys_2d = [k for k in filtered_keys if task_vectors[0][k].dim() == 2]
    keys_1d = [k for k in filtered_keys if task_vectors[0][k].dim() < 2]
    keys_nd = [k for k in filtered_keys if task_vectors[0][k].dim() > 2]

    merged_tv = {}
    total_common = 0
    total_unique = 0
    total_dims = 0
    n_bimodal = 0
    start_time = time.time()

    for idx, key in enumerate(keys_2d):
        tvs = [task_vectors[i][key] for i in range(K)]

        merged, stats = cusd_merge_layer(
            tvs, method_fn, method_kwargs,
            pi=pi, breakpoint_method=breakpoint_method,
            beta=beta, device=device,
            classify_mode=classify_mode,
            fallback_quantile=fallback_quantile,
            fixed_threshold=fixed_threshold,
        )

        merged_tv[key] = merged

        n = stats.get("n", 0)
        total_dims += n
        total_common += stats.get("n_common", 0)
        total_unique += stats.get("n_unique", 0)
        if stats.get("is_bimodal"):
            n_bimodal += 1

        if idx % log_interval == 0 or idx == len(keys_2d) - 1:
            elapsed = time.time() - start_time
            unique_frac = total_unique / total_dims if total_dims > 0 else 0
            print(f"  Layer {idx+1}/{len(keys_2d)}: "
                  f"common={total_common}, unique={total_unique} ({unique_frac:.1%}), "
                  f"elapsed={elapsed:.1f}s", flush=True)

        gc.collect()

    # 1D parameters: apply method directly
    for key in keys_1d:
        tvs = [task_vectors[i][key] for i in range(K)]
        merged_tv[key] = method_fn(tvs, **method_kwargs)

    # >2D parameters: apply method directly
    for key in keys_nd:
        tvs = [task_vectors[i][key] for i in range(K)]
        merged_tv[key] = method_fn(tvs, **method_kwargs)

    elapsed = time.time() - start_time
    common_frac = total_common / total_dims if total_dims > 0 else 0
    unique_frac = total_unique / total_dims if total_dims > 0 else 0

    merge_stats = {
        "method": "CUSD",
        "pi": pi,
        "breakpoint_method": breakpoint_method,
        "beta": beta,
        "classify_mode": classify_mode,
        "n_2d_layers": len(keys_2d),
        "n_1d_params": len(keys_1d),
        "n_nd_params": len(keys_nd),
        "total_directions": total_dims,
        "total_common": total_common,
        "total_unique": total_unique,
        "common_fraction": round(common_frac, 4),
        "unique_fraction": round(unique_frac, 4),
        "n_bimodal": n_bimodal,
        "n_non_bimodal": len(keys_2d) - n_bimodal,
        "elapsed_sec": round(elapsed, 1),
    }

    return merged_tv, merge_stats


def plain_merge_full(
    task_vectors: List[Dict[str, torch.Tensor]],
    method_fn: Callable,
    method_kwargs: dict,
    skip_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    Plain merge without CUSD decomposition (baseline).

    Args:
        task_vectors: list of K dicts {key: tensor}.
        method_fn: merge function.
        method_kwargs: kwargs for method.
        skip_keys: key patterns to skip.

    Returns:
        merged_tv: dict {key: tensor}.
        merge_stats: dict.
    """
    if skip_keys is None:
        skip_keys = ["embed_tokens", "lm_head"]

    K = len(task_vectors)
    all_keys = sorted(task_vectors[0].keys())
    filtered_keys = [
        k for k in all_keys
        if not any(skip in k for skip in skip_keys)
    ]

    merged_tv = {}
    start_time = time.time()

    for key in filtered_keys:
        tvs = [task_vectors[i][key] for i in range(K)]
        merged_tv[key] = method_fn(tvs, **method_kwargs)

    elapsed = time.time() - start_time
    merge_stats = {
        "method": "plain",
        "n_keys": len(filtered_keys),
        "elapsed_sec": round(elapsed, 1),
    }

    return merged_tv, merge_stats
