import os
import sys

import numpy as np
import torch
from torch import nn

import config
from lib.ml_tasks import matching
from lib.models.abstract_model import AbstractModel
from lib.torch_models import dssm


class FaceFashionItemMatching(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.model = self.load_base_model()
        self.model.eval()

    def apply_batched(self, input):
        '''
        input -- list of np arrays
        '''
        if len(input) == 0:
            return []

        with torch.no_grad():
            output = [
                t.detach().cpu().item() for t in self.model(
                    torch.stack([torch.tensor(t).float() for t in input['x']]),
                    torch.stack([torch.tensor(t).float() for t in input['y']]),
                )
            ]

        return output

    def apply(self, input):
        with torch.no_grad():
            return self.model(
                torch.tensor(input['x']).float().unsqueeze(0),
                torch.tensor(input['y']).float().unsqueeze(0),
            )[0].item()

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
        return matching.Matching.load_from_checkpoint(
            os.path.join(
                os.environ['PROJECT_DIR'],
                'data',
                'checkpoints',
                'dssm_vgg',
                'checkpoints',
                'epoch=34-step=7033.ckpt',
            ),
            model=dssm.DSSM(x_arch=[2622, 512, 128], y_arch=[512, 256, 128], p_dropout=0.5),
        )
