"""
Task Arithmetic (TA) merging.
"""
import torch
from typing import List


def merge(
    task_vectors: List[torch.Tensor],
    scaling: float = 0.4,
    **kwargs,
) -> torch.Tensor:
    """
    Task Arithmetic: scaled sum of task vectors.

    Args:
        task_vectors: list of K tensors, all same shape.
        scaling: scaling factor (lambda in paper).

    Returns:
        merged tensor.
    """
    result = torch.zeros_like(task_vectors[0])
    for tv in task_vectors:
        result += tv
    return result * scaling
