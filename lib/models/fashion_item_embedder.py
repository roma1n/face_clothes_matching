import os

import numpy as np
import torch
from torch import nn

import config
from lib.ml_tasks import multi_head_classification
from lib.models.abstract_model import AbstractModel
from lib.torch_models import resnet


class FashionItemEmbedder(AbstractModel, nn.Module):
    def __init__(self):
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.model = self.load_base_model()

    def apply(self, input):
        if isinstance(input, str):
            input = np.load(input)
        return self.model(torch.tensor(input).float().unsqueeze(0))[0].tolist()

    def get_torch_model(self):
        '''
        Returns torch model if exists. Returns None by default.
        '''
        return None

    def train(self):
        '''
        Runs model training if needed.
        '''
        pass

    def load_base_model(self):
        heads_desc = {
            'category': 56,
            'season': 4,
            'color': 20,
            'print': 11,
            # 'country': 73,
            # 'brand': 652,
        }

        embedder = resnet.ExtendedResnetEmbedder(n_extend_chennels=0)

        return multi_head_classification.MultiHeadClassification.load_from_checkpoint(
            os.path.join(
                os.environ['PROJECT_DIR'],
                'data',
                'checkpoints',
                'resnet_prevent_overfitting',
                'checkpoints',
                'epoch=4-step=904.ckpt',
            ),
            embedder=embedder,
            heads_desc=heads_desc,
        )
