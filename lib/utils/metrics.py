import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F


def intersection_over_union(prediction: torch.tensor, mask: torch.tensor, threshold: float = 0.1):
    intersection = (prediction > threshold) & (mask > threshold)
    union = (prediction > threshold) | (mask > threshold)

    return intersection.sum() / union.sum()


def roc_auc(logits: torch.tensor, labels: torch.tensor, num_classes: int):
    probs_np = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    labels_np = np.argmax(labels_np, axis=1)

    ret = metrics.roc_auc_score(
        labels_np,
        probs_np,
        average='macro',
        multi_class='ovo',
        labels=np.arange(num_classes),
    )

    if np.isnan(ret):
        ret = 0.0

    return ret


def dssm_roc_auc(logits):
    if len(logits) < 2:  # Only one object -> only positives
        return 1.0

    probs_np = F.softmax(logits, dim=1).detach().cpu().numpy()
    positives = probs_np[:, 0].flatten()
    negatives = probs_np[:, 1:].flatten()

    return metrics.roc_auc_score(
        y_true=np.concatenate([
            np.ones(positives.shape),
            np.zeros(negatives.shape),
        ]),
        y_score=np.concatenate([positives, negatives]),
    )


def accuracy(logits: torch.tensor, labels: torch.tensor):
    probs_np = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    return metrics.accuracy_score(
        np.argmax(labels_np, axis=1),
        np.argmax(probs_np, axis=1),
    )
