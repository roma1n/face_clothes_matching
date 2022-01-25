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

from lib import metrics
from lib.torch_models import resnet


class MultiHeadClassification(LightningModule):
    def __init__(
        self,
        embedder: nn.Module,
        heads_desc: Dict[str, int],
        embedder_lr: float = 1e-4,
        heads_lr: float = 1e-3,
        embedding_size: int = 512,
    ):
        super().__init__()

        self.embedder = embedder
        self.embedder_lr = embedder_lr
        self.heads_lr = heads_lr
        self.heads_desc = heads_desc
        self.embedding_size = embedding_size

        self.heads = nn.ModuleDict({
            head_name: nn.Linear(
                self.embedding_size,
                self.heads_desc[head_name],
            ) for head_name in self.heads_desc
        })
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.embedder(x)

    def step(self, batch, batch_idx):
        image = batch['img']
        labels = {k: v for k, v in batch.items() if k != 'img'}

        embedding = self.embedder(image)
        logits = {head_name: self.heads[head_name](embedding) for head_name in labels}
        loss = None
        for head_name in labels:
            head_loss = self.criterion(logits[head_name], labels[head_name])
            if loss is None:
                loss = head_loss
            else:
                loss += head_loss

        return loss, {"loss": loss.item()}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.embedder.parameters(), 'lr': self.embedder_lr},
            {'params': self.heads.parameters(), 'lr': self.heads_lr},
        ])
 

class ClassificationDataset(Dataset):
    def __init__(
        self,
        path,
        desc,
        heads_desc,
        split='train',
        id_col='id',
      ):
        super().__init__()

        self.path = path
        self.desc = desc
        self.heads_desc = heads_desc
        self.split = split
        self.id_col = id_col

        files = os.listdir(self.path)

        self.desc = list(filter(
            lambda x: '{}.npy'.format(x['img']) in files,
            self.desc,
        ))

        self.encoders = {
            k: preprocessing.LabelEncoder().fit(
                list(map(lambda x: x[k], self.desc))
            ) for k in self.heads_desc
        }

        if self.split == 'train':
            self.desc = list(filter(
                lambda x: x[self.id_col] % 10 >= 2,
                self.desc,
            ))
        elif self.split == 'val':
            self.desc = list(filter(
                lambda x: x[self.id_col] % 10 == 1,
                self.desc,
            ))
        elif self.split == 'test':
            self.desc = list(filter(
                lambda x: x[self.id_col] % 10 == 0,
                self.desc,
            ))

    def __len__(self):
        return len(self.desc)

    def _get_label(self, head, value):
        if value is None:
            res = np.ones(self.heads_desc[head], dtype=float) / self.heads_desc[head]
        else:
            res = np.zeros(self.heads_desc[head], dtype=float)
            res[self.encoders[head].transform([value])[0]] = 1.

        return torch.tensor(res).float()


    def __getitem__(self, idx):
        item_desc = self.desc[idx]
        fname = item_desc['img']

        item = {'img': torch.tensor(np.load(os.path.join(self.path, '{}.npy'.format(fname)))).float()}

        item.update({
            k: self._get_label(k, item_desc[k] if k in item_desc else None) for k in self.heads_desc
        })

        return item




class FaceClassificationDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.images_path = os.path.join(self.data_dir, 'transformed')

        self.train_dataset = ClassificationDataset(self.images_path, self.desc, split='train')
        self.val_dataset = ClassificationDataset(self.images_path, self.desc, split='val')
        self.test_dataset = ClassificationDataset(self.images_path, self.desc, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class ClothesClassificationDataModule(LightningDataModule):
    '''
    Desc example:
    {
        'id': 708,
        'link': 'https://www.lamoda.ru/p/mp002xm1gzy3/clothes-northland-bryuki-gornolyzhnye/',
        'img': 'MP002XM1GZY3_13289924_1_v1_2x.jpg',
        'category': 'Брюки',
        'season': 'демисезон',
        'color': 'черный',
        'print': 'однотонный',
        'country': 'Вьетнам',
        'brand': 'Northland'
    }
    '''
    def __init__(self, data_dir: str, batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.images_path = os.path.join(self.data_dir, 'transformed')

        self.heads_desc = {
            'category': 56,
            'season': 4,
            'color': 20,
            'print': 11,
            # 'country': 73,
            # 'brand': 652,
        }

        with open(os.path.join(self.data_dir, 'desc.json'), 'r') as f:
            self.desc = json.loads(f.read())

        self.train_dataset = ClassificationDataset(self.images_path, self.desc, self.heads_desc, split='train')
        self.val_dataset = ClassificationDataset(self.images_path, self.desc, self.heads_desc, split='val')
        self.test_dataset = ClassificationDataset(self.images_path, self.desc, self.heads_desc, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


def main():
    dm = ClothesClassificationDataModule(os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda'))
    ml_task = MultiHeadClassification(resnet.ExtendedResnetEmbedder(n_extend_chennels=0), dm.heads_desc)
    trainer = Trainer()

    trainer.fit(ml_task, datamodule=dm)
    return dm, ml_task, trainer


if __name__ == "__main__":
    dm, ml_task, trainer = main()
