import torch


def intersection_over_union(prediction: torch.tensor, mask: torch.tensor, threshold: float = 0.1):
    intersection = (prediction > threshold) & (mask > threshold)
    union = (prediction > threshold) | (mask > threshold)

    return intersection.sum() / union.sum()
