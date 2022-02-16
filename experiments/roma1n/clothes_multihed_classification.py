import os

import pytorch_lightning
from pytorch_lightning.callbacks import early_stopping

from lib.torch_models import resnet
from lib.ml_tasks import multi_head_classification


def main():
    dm = multi_head_classification.ClothesClassificationDataModule(os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda'))
    ml_task = multi_head_classification.MultiHeadClassification(resnet.ExtendedResnetEmbedder(n_extend_chennels=0), dm.heads_desc)
    trainer = pytorch_lightning.Trainer(
        callbacks=[
            early_stopping.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=3,
                mode='min',
            ),
        ],
        val_check_interval=0.25,  # check validation score 4 times per epoch
    )

    trainer.fit(ml_task, datamodule=dm)
    return dm, ml_task, trainer


if __name__ == '__main__':
    dm, ml_task, trainer = main()
