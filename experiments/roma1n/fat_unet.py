import os

import pytorch_lightning

from lib.torch_models import unet
from lib.ml_tasks import segmentation


def main():
    dm = segmentation.SegmentationDataModule(
        os.path.join(os.environ['PROJECT_DIR'], 'data', 'segmentation'),
        full=False,
    )

    model = unet.UNET(num_classes=dm.num_classes, base_channel_num=16)

    ml_task = segmentation.Segmenation(model)
    trainer = pytorch_lightning.Trainer()

    trainer.fit(ml_task, datamodule=dm)
    return dm, ml_task, trainer


if __name__ == '__main__':
    dm, ml_task, trainer = main()
