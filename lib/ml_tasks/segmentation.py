import json
import numpy as np
import os

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

from lib.utils import metrics
from lib.torch_models import (
    unet,
    unet_resnet,
    unet_transformer,
)


class Segmenation(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        image, mask = batch

        logits = self.model(image)
        loss = self.criterion(logits, mask)
        iou = metrics.intersection_over_union(logits, mask)

        return loss, {"loss": loss.item(), 'iou': iou}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log('train_step_iou', logs['iou'], on_step=True, prog_bar=True)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
 

class SegmentationDataset(Dataset):
    def __init__(
        self,
        path,
        desc,
        split='train',
      ):
        super().__init__()

        self.path = path
        self.desc = desc
        self.split = split

        self.cid_to_layer = {'0': 0, '1': 1, '4': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, '21': 8, '23': 9}

        files = os.listdir(path)

        self.desc = list(filter(
            lambda x: '{}.npy'.format(x['num']) in files,
            self.desc,
        ))

        if self.split == 'train':
            self.desc = list(filter(
                lambda x: x['num'] % 10 >= 2,
                self.desc,
            ))
        elif self.split == 'val':
            self.desc = list(filter(
                lambda x: x['num'] % 10 == 1,
                self.desc,
            ))
        elif self.split == 'test':
            self.desc = list(filter(
                lambda x: x['num'] % 10 == 0,
                self.desc,
            ))

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        arr = np.load(os.path.join(self.path, '{}.npy'.format(self.desc[idx]['num'])))

        cids = self.desc[idx]['cids']

        image = arr[:3]
        mask = np.zeros((10, 224, 224))
        
        for cid in cids:
            mask[self.cid_to_layer[cid]] = arr[3 + cids[cid]]

        return torch.tensor(image).float(), torch.tensor(mask).float()



class FullSegmentationDataset(Dataset):
    def __init__(
        self,
        path,
        desc,
        split='train',
      ):
        super().__init__()

        self.path = path
        self.desc = desc

        files = os.listdir(path)

        self.desc = list(filter(
            lambda x: '{}.npy'.format(x['num']) in files,
            self.desc,
        ))

        if split == 'train':
            self.desc = list(filter(
                lambda x: x['num'] % 10 >= 2,
                self.desc,
            ))
        elif split == 'val':
            self.desc = list(filter(
                lambda x: x['num'] % 10 == 1,
                self.desc,
            ))
        elif split == 'test':
            self.desc = list(filter(
                lambda x: x['num'] % 10 == 0,
                self.desc,
            ))

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        arr = np.load(os.path.join(self.path, '{}.npy'.format(self.desc[idx]['num'])))

        cids = self.desc[idx]['cids']

        image = arr[:3]
        mask = np.zeros((46, 224, 224))
        mask[cids] = arr[3:]

        return torch.tensor(image).float(), torch.tensor(mask).float()


class SegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        full: bool = False,
        batch_size: int = 100,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.full = full
        self.batch_size = batch_size

        if self.full:
            with open(os.path.join(self.data_dir, 'desc.json'), 'r') as f:
                self.desc = json.loads(f.read())
            self.DatasetType = FullSegmentationDataset
            self.num_classes = 46
        else:
            with open(os.path.join(self.data_dir, 'desc_short.json'), 'r') as f:
                self.desc = json.loads(f.read())
            self.DatasetType = SegmentationDataset
            self.num_classes = 10

        self.images_path = os.path.join(self.data_dir, 'transformed')

        self.dataset_splits = {
            split: self.DatasetType(
                self.images_path,
                self.desc,
                split=split,
            ) for split in ['train', 'val', 'test']
        }

    def _dataloader_by_split(self, split):
        return DataLoader(
            self.dataset_splits[split],
            batch_size=self.batch_size,
            shuffle=True if split == 'train' else False,
        )

    def train_dataloader(self):
        return self._dataloader_by_split('train')

    def val_dataloader(self):
        return self._dataloader_by_split('val')

    def test_dataloader(self):
        return self._dataloader_by_split('test')


def main():
    dm = SegmentationDataModule(os.path.join(os.environ['PROJECT_DIR'], 'data', 'segmentation'), full=False)

    # model = unet_transformer.UnetTransformer(num_classes=dm.num_classes)
    model = unet.UNET(num_classes=dm.num_classes, base_channel_num=16)
    # ckpt = torch.load('lightning_logs/version_6/checkpoints/epoch=0-step=299.ckpt')
    # model.load_state_dict({
    #     '.'.join(k.split('.')[1:]): v for k, v in ckpt['state_dict'].items()
    # })

    ml_task = Segmenation(model)
    trainer = Trainer()

    trainer.fit(ml_task, datamodule=dm)
    return dm, ml_task, trainer


if __name__ == "__main__":
    dm, ml_task, trainer = main()
