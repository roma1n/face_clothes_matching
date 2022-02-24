import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.transform as sktransform

import torch

import config
from lib.ml_tasks import segmentation
from lib.models.abstract_model import AbstractModel
from lib.torch_models import unet


class FashionItemSegmentation(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)

        self.model = self.load_base_model()

    def apply(self, input):
        pass

    def apply_batched(self, input, preprocessed=False, return_input=False):
        '''
        input -- list of np arrays
        '''
        if len(input) == 0:
            return []

        if isinstance(input[0], str):
            input = [np.load(i) if preprocessed else plt.imread(i) for i in input]

        if not preprocessed:
            input = [self.preprocess(i) for i in input]

        output = [
            t.detach().cpu().numpy() for t in self.model(
                torch.stack([torch.tensor(t).float() for t in input])
            )
        ]

        if not return_input:
            return output
        else:
            return output, input

    def get_torch_model(self):
        '''
        Returns torch model if exists. Returns None by default.
        '''
        return self.model

    def train(self):
        '''
        Runs model training if needed.
        '''
        pass

    def load_base_model(self):
        return segmentation.Segmenation.load_from_checkpoint(
            os.path.join(
                os.environ['PROJECT_DIR'],
                'data',
                'checkpoints',
                'vanila_unet_15ep_31iou',  # 'fat_unet_2_weeks',
                'checkpoints',
                'epoch=13-step=4199.ckpt',  # 'epoch=123-step=37199.ckpt',
            ),
            model=unet.UNET(
                num_classes=10,
                base_channel_num=8,  # 16,
            ),
        )

    def preprocess(self, input):
        return sktransform.resize(input, config.EMBEDDER_INPUT_SHAPE).transpose(2, 0, 1)
