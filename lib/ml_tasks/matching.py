import json
import numpy as np
import os
from sklearn import preprocessing
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from lib.utils import metrics
from lib.torch_models import dssm


class Matching(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.model = model
        self.lr = lr

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.embedder(x)

    def step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']

        batch_size = x.shape[0]

        # Predict score for each pair x-y in batch. All other elements in batch are negatives
        logits = torch.stack([
            self.model(x, torch.roll(y, i, 0)) for i in range(batch_size)
        ], dim=1)

        loss = self.criterion(
            logits,
            torch.zeros(batch_size).long(),  # The truth pair for each object is in row with zero shift
        )

        logs = {
            'loss': loss.item(),
            'roc_auc': metrics.dssm_roc_auc(logits)
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'step_train_{k}': v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f'val_{k}': v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr, 'weight_decay': 1e-4},
        ])
 

class MatchingDataset(Dataset):
    def __init__(
        self,
        desc,
        split='train',
        x_col='x',
        y_col='y',
        id_col='fname',
      ):
        super().__init__()

        self.desc = desc
        self.split = split
        self.x_col = x_col
        self.y_col = y_col
        self.id_col = id_col

        if self.split == 'train':
            self.desc = list(filter(
                lambda elem: hash(elem[self.id_col]) % 10 >= 2,
                self.desc,
            ))
        elif self.split == 'val':
            self.desc = list(filter(
                lambda elem: hash(elem[self.id_col]) % 10 == 1,
                self.desc,
            ))
        elif self.split == 'test':
            self.desc = list(filter(
                lambda elem: hash(elem[self.id_col]) % 10 == 0,
                self.desc,
            ))

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        item_desc = self.desc[idx]

        return {
            'x': torch.tensor(item_desc[self.x_col]).float(),
            'y': torch.tensor(item_desc[self.y_col]).float(),
        }


class FaceClothesMatchingDataModule(LightningDataModule):
    '''
    Desc example:
    {
        'fname': 'xxx.jpg',
        'face': [0.0, 0.2, -0.3],
        'fashion_item': [0.4, -0.1, -0.7],
    }
    '''
    def __init__(self, data_dir: str, batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        with open(os.path.join(self.data_dir, 'embeddings.json'), 'r') as f:
            self.desc = json.loads(f.read())

        self.x_col = 'face'
        self.y_col = 'fashion_item'
        self.id_col = 'fname'

        self.train_dataset = MatchingDataset(
            self.desc,
            split='train',
            x_col=self.x_col,
            y_col=self.y_col,
            id_col=self.id_col,
        )
        self.val_dataset = MatchingDataset(
            self.desc,
            split='val',
            x_col=self.x_col,
            y_col=self.y_col,
            id_col=self.id_col,
        )
        self.test_dataset = MatchingDataset(
            self.desc,
            split='test',
            x_col=self.x_col,
            y_col=self.y_col,
            id_col=self.id_col,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
