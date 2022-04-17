import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.transform as sktransform

import torch
from torch import nn

import config
from lib.ml_tasks import multi_head_classification
from lib.models.abstract_model import AbstractModel
from lib.torch_models import resnet


class FashionItemEmbedder(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.model = self.load_base_model()
        self.model.eval()

    def apply(self, input, preprocessed=False):
        if isinstance(input, str):
            input = np.load(input) if preprocessed else plt.imread(input)
        if not preprocessed:
            input = self.preprocess(input)
        with torch.no_grad():
            return self.model(torch.tensor(input).float().unsqueeze(0))[0].tolist()

    def apply_batched(self, input, preprocessed=False):
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

        return output

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
                'resnet_no_faces',
                'checkpoints',
                'epoch=6-step=1315.ckpt',
            ),
            embedder=embedder,
            heads_desc=heads_desc,
        )

    def preprocess(self, input):
        return sktransform.resize(input, config.EMBEDDER_INPUT_SHAPE).transpose(2, 0, 1)
