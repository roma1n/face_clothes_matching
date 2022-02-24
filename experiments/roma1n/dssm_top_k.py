import os

import pytorch_lightning
from pytorch_lightning.callbacks import early_stopping

from lib.torch_models import dssm
from lib.ml_tasks import matching


def main():
    dm = matching.FaceClothesMatchingDataModule(os.path.join(os.environ['PROJECT_DIR'], 'data', 'lamoda'))
    # model = dssm.DSSM(x_arch=[512, 256, 128], y_arch=[512, 256, 128], p_dropout=0.2)
    ml_task = matching.Matching.load_from_checkpoint(
        'lightning_logs/version_2/checkpoints/epoch=8-step=1657.ckpt',
        model=dssm.DSSM(x_arch=[512, 256, 128], y_arch=[512, 256, 128], p_dropout=0.2),
        top_k=10,
    )

    # Pretraining
    # ml_task = matching.Matching(model=model)
    # trainer = pytorch_lightning.Trainer(
    #     callbacks=[
    #         early_stopping.EarlyStopping(
    #             monitor='val_roc_auc',
    #             min_delta=0.0,
    #             patience=5,
    #             mode='max',
    #         ),
    #     ],
    #     val_check_interval=0.25,  # check validation score 4 times per epoch
    # )
    # trainer.fit(ml_task, datamodule=dm)

    # Training with top k
    trainer = pytorch_lightning.Trainer(
        callbacks=[
            early_stopping.EarlyStopping(
                monitor='val_roc_auc',
                min_delta=0.0,
                patience=5,
                mode='max',
            ),
        ],
        val_check_interval=0.25,  # check validation score 4 times per epoch
    )
    trainer.fit(ml_task, datamodule=dm)

    return dm, ml_task, trainer


if __name__ == '__main__':
    dm, ml_task, trainer = main()
