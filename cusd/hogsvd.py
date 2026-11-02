"""
HO-GSVD computation and KDE-based direction classification.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin


def hogsvd(
    matrices: List[torch.Tensor],
    pi: float = 1e-3,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Dict]:
    """
    Higher-Order Generalized SVD.

    Args:
        matrices: list of K matrices [A_1, ..., A_K], each (m_i, n), float32.
                  All must share the same column dimension n.
        pi: Tikhonov regularization for Gram matrices.
        device: compute device ("cuda" or "cpu").

    Returns:
        V: shared right basis (n, n), orthonormal columns, descending eigenvalue order.
        eigenvalues: (n,) eigenvalues of S, descending.
        Sigma_list: list of K tensors, each (n,), directional energies ||A_i V_k||.
        diagnostics: dict with S asymmetry norm.
    """
    K = len(matrices)
    assert K >= 2, f"Need at least 2 matrices, got {K}"

    n = matrices[0].shape[1]
    for i, A in enumerate(matrices):
        assert A.shape[1] == n, f"Matrix {i} has {A.shape[1]} columns, expected {n}"

    # Regularized Gram matrices (float64 for numerical stability)
    eye_n = torch.eye(n, device=device, dtype=torch.float64)
    gram = []
    for A in matrices:
        A_dev = A.to(device=device, dtype=torch.float64)
        G = A_dev.T @ A_dev + pi * eye_n
        gram.append(G)
        del A_dev
        if device == "cuda":
            torch.cuda.empty_cache()

    # Pairwise quotient matrix
    S = torch.zeros((n, n), device=device, dtype=torch.float64)
    for i in range(K):
        for j in range(i + 1, K):
            S += torch.linalg.solve(gram[j], gram[i])
            S += torch.linalg.solve(gram[i], gram[j])
    S /= (K * (K - 1))

    # Symmetrize (S is not guaranteed symmetric)
    asym_norm = torch.norm(S - S.T).item() / (torch.norm(S).item() + 1e-12)
    S = (S + S.T) / 2

    # Eigendecomposition
    eigenvalues, V = torch.linalg.eigh(S)
    eigenvalues = eigenvalues.flip(0)
    V = V.flip(1)

    V = V.float()
    eigenvalues = eigenvalues.float()

    # Directional energies: sigma_ik = ||A_i V_k||
    Sigma_list = []
    for A in matrices:
        A_dev = A.to(device=device, dtype=torch.float32)
        B = A_dev @ V
        sigmas = torch.linalg.norm(B, dim=0)
        Sigma_list.append(sigmas.cpu())
        del A_dev, B
        if device == "cuda":
            torch.cuda.empty_cache()

    del gram, S
    if device == "cuda":
        torch.cuda.empty_cache()

    return V.cpu(), eigenvalues.cpu(), Sigma_list, {"asym_norm": asym_norm}


def compute_sigma_ratios(
    Sigma_list: List[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean-normalized ratios and per-direction dispersion.

    For each direction k:
      r_ik = sigma_ik / mean_j(sigma_jk)
      std_k = std_i(r_ik)

    Returns:
        r_matrix: (K, n) mean-normalized ratios.
        std_per_direction: (n,) dispersion across experts per direction.
        sigma_matrix: (K, n) raw sigma values.
        mean_sigma: (n,) mean sigma per direction.
    """
    K = len(Sigma_list)
    n = len(Sigma_list[0])

    sigma_matrix = np.zeros((K, n), dtype=np.float64)
    for i, sig in enumerate(Sigma_list):
        sigma_matrix[i] = sig.numpy().astype(np.float64)

    mean_sigma = sigma_matrix.mean(axis=0)
    mean_sigma_safe = np.where(mean_sigma > 1e-12, mean_sigma, 1e-12)

    r_matrix = sigma_matrix / mean_sigma_safe[np.newaxis, :]
    std_per_direction = r_matrix.std(axis=0)

    return r_matrix, std_per_direction, sigma_matrix, mean_sigma


def find_breakpoint_kde(
    std_values: np.ndarray,
    bandwidth: str = "scott",
    n_eval: int = 1000,
    fallback_quantile: float = 0.9,
) -> Tuple[float, Dict]:
    """
    Find the natural breakpoint in dispersion values via KDE valley detection.

    If bimodal, the valley between modes separates common from unique.
    If unimodal, returns a reference quantile (caller treats all as common).

    Returns:
        theta: breakpoint threshold.
        info: dict with is_bimodal flag and diagnostics.
    """
    valid = std_values[std_values > 1e-8]
    if len(valid) < 10:
        theta = float(np.median(std_values))
        return theta, {"is_bimodal": False, "reason": "too_few_valid_points", "theta": theta}

    try:
        kde = gaussian_kde(valid, bw_method=bandwidth)
    except Exception:
        theta = float(np.quantile(std_values, fallback_quantile))
        return theta, {"is_bimodal": False, "reason": "kde_failed", "theta": theta}

    x_eval = np.linspace(valid.min(), valid.max(), n_eval)
    y_eval = kde(x_eval)

    minima_indices = argrelmin(y_eval, order=max(5, n_eval // 50))[0]

    if len(minima_indices) > 0:
        mid = (valid.min() + valid.max()) / 2
        best_idx = minima_indices[np.argmin(np.abs(x_eval[minima_indices] - mid))]
        theta = float(x_eval[best_idx])
        return theta, {
            "is_bimodal": True,
            "theta": theta,
            "valley_x": float(x_eval[best_idx]),
            "valley_y": float(y_eval[best_idx]),
            "n_valleys": len(minima_indices),
        }
    else:
        theta = float(np.quantile(std_values, fallback_quantile))
        return theta, {
            "is_bimodal": False,
            "reason": "no_valley_found",
            "theta": theta,
            "fallback_quantile": fallback_quantile,
        }


def find_breakpoint_gmm(
    std_values: np.ndarray,
    fallback_quantile: float = 0.9,
) -> Tuple[float, Dict]:
    """
    Find breakpoint via GMM (k=2). Crossover between components is the threshold.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        theta = float(np.quantile(std_values, fallback_quantile))
        return theta, {"is_bimodal": False, "reason": "sklearn_not_available", "theta": theta}

    valid = std_values[std_values > 1e-8].reshape(-1, 1)
    if len(valid) < 20:
        theta = float(np.quantile(std_values, fallback_quantile))
        return theta, {"is_bimodal": False, "reason": "too_few_points", "theta": theta}

    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(valid)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        order = np.argsort(means)
        mu0, mu1 = means[order[0]], means[order[1]]
        std0, std1 = stds[order[0]], stds[order[1]]

        if std0 + std1 > 1e-10:
            theta = (mu0 * std1 + mu1 * std0) / (std0 + std1)
        else:
            theta = (mu0 + mu1) / 2

        separation = abs(mu1 - mu0) / (std0 + std1 + 1e-10)
        is_bimodal = separation > 0.5

        if not is_bimodal:
            theta = float(np.quantile(std_values, fallback_quantile))

        return float(theta), {
            "is_bimodal": is_bimodal,
            "theta": float(theta),
            "gmm_means": [float(mu0), float(mu1)],
            "gmm_stds": [float(std0), float(std1)],
            "separation": float(separation),
        }
    except Exception as e:
        theta = float(np.quantile(std_values, fallback_quantile))
        return theta, {"is_bimodal": False, "reason": f"gmm_failed: {e}", "theta": theta}


CLASSIFY_MODES = [
    "rel_std",
    "abs_std",
]


def classify_subspaces(
    Sigma_list: List[torch.Tensor],
    breakpoint_method: str = "kde",
    fallback_quantile: float = 0.9,
    classify_mode: str = "abs_std",
    fixed_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Classify HO-GSVD directions as common or unique.

    classify_mode controls the dispersion scoring:
      - "abs_std": std of raw sigma values (absolute dispersion, default).
      - "rel_std": std of mean-normalized ratios r_ik (relative dispersion).

    When KDE finds no valley (unimodal distribution), all directions are
    classified as common (the layer has no clearly separable unique structure).

    Args:
        Sigma_list: list of K tensors from hogsvd, each (n,).
        breakpoint_method: "kde", "gmm", or "fixed_value".
        fallback_quantile: quantile for reference when no valley found.
        classify_mode: dispersion scoring strategy.
        fixed_threshold: threshold when breakpoint_method="fixed_value".

    Returns:
        common_mask: (n,) boolean, True = common direction.
        unique_mask: (n,) boolean, True = unique direction.
        theta: breakpoint threshold.
        diagnostics: dict with classification details.
    """
    r_matrix, std_per_direction, sigma_matrix, mean_sigma = compute_sigma_ratios(Sigma_list)

    K = len(Sigma_list)
    n = len(Sigma_list[0])

    # Compute classify values based on mode
    if classify_mode == "rel_std":
        classify_values = std_per_direction

    elif classify_mode == "abs_std":
        classify_values = sigma_matrix.std(axis=0)

    else:
        raise ValueError(f"Unknown classify_mode: {classify_mode}. Choose from {CLASSIFY_MODES}")

    # Find breakpoint
    if breakpoint_method == "fixed_value":
        if fixed_threshold is None:
            raise ValueError("fixed_threshold required when breakpoint_method='fixed_value'")
        theta = float(fixed_threshold)
        bp_info = {"is_bimodal": True, "reason": "fixed_value", "theta": theta}
    elif breakpoint_method == "kde":
        theta, bp_info = find_breakpoint_kde(classify_values, fallback_quantile=fallback_quantile)
    elif breakpoint_method == "gmm":
        theta, bp_info = find_breakpoint_gmm(classify_values, fallback_quantile=fallback_quantile)
    else:
        raise ValueError(f"Unknown breakpoint method: {breakpoint_method}")

    # When no valley found (unimodal): treat all directions as common
    non_bimodal_all_common = (
        breakpoint_method in ("kde", "gmm")
        and not bp_info.get("is_bimodal", False)
    )

    if non_bimodal_all_common:
        common_mask = np.ones(n, dtype=bool)
        unique_mask = np.zeros(n, dtype=bool)
    else:
        common_mask = classify_values < theta
        unique_mask = ~common_mask

    # Diagnostics
    common_energy = float(mean_sigma[common_mask].sum()) if common_mask.any() else 0.0
    unique_energy = float(mean_sigma[unique_mask].sum()) if unique_mask.any() else 0.0
    total_energy = float(mean_sigma.sum())

    diagnostics = {
        "K": K,
        "n_directions": n,
        "n_common": int(common_mask.sum()),
        "n_unique": int(unique_mask.sum()),
        "common_fraction": float(common_mask.sum() / n) if n > 0 else 0.0,
        "theta": float(theta),
        "classify_mode": classify_mode,
        "breakpoint_info": bp_info,
        "breakpoint_reason": bp_info.get("reason"),
        "non_bimodal_all_common": bool(non_bimodal_all_common),
        "r_matrix": r_matrix,
        "std_per_direction": std_per_direction,
        "mean_sigma_per_direction": mean_sigma,
        "energy_total": round(total_energy, 4),
        "energy_common": round(common_energy, 4),
        "energy_unique": round(unique_energy, 4),
    }

    return common_mask, unique_mask, theta, diagnostics


def decompose_task_vectors(
    task_vectors: List[torch.Tensor],
    V: torch.Tensor,
    common_mask: np.ndarray,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Project task vectors into common and unique components.

    C_i = T_i @ V_common @ V_common^T  (projection onto common subspace)
    R_i = T_i - C_i                     (unique residual)

    V is orthonormal from eigh, so projection is idempotent.

    Args:
        task_vectors: list of K matrices, each (m, n).
        V: full right basis from hogsvd, shape (n, n).
        common_mask: boolean array (n,), True = common direction.

    Returns:
        C_list: list of common components.
        R_list: list of unique residuals.
    """
    V_common = V[:, torch.from_numpy(common_mask)]

    C_list = []
    R_list = []
    for T_i in task_vectors:
        C_i = (T_i @ V_common) @ V_common.T
        R_i = T_i - C_i
        C_list.append(C_i)
        R_list.append(R_i)

    return C_list, R_list
