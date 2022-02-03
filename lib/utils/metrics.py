import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F


def intersection_over_union(prediction: torch.tensor, mask: torch.tensor, threshold: float = 0.1):
    intersection = (prediction > threshold) & (mask > threshold)
    union = (prediction > threshold) | (mask > threshold)

    return intersection.sum() / union.sum()


def roc_auc(logits: torch.tensor, labels: torch.tensor, num_classes: int):
    predictions_np = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    labels_np = np.argmax(labels_np, axis=1)

    ret = metrics.roc_auc_score(
        labels_np,
        predictions_np,
        average='macro',
        multi_class='ovo',
        labels=np.arange(num_classes),
    )

    if np.isnan(ret):
        ret = 0.0

    return ret


def accuracy(logits: torch.tensor, labels: torch.tensor):
    predictions_np = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    return metrics.accuracy_score(
        np.argmax(labels_np, axis=1),
        np.argmax(predictions_np, axis=1),
    )
