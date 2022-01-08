import torch
from torch import nn

import config
from lib.models.abstract_model import AbstractModel
from lib.models.embedder_trainer import EmbedderTrainer


class FaceEmbedder(AbstractModel, nn.Module):
    def __init__(self):
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.model = self.load_base_model()

    def apply(self, input):
        pass

    def apply_batched(self, batch):
        '''
        Model application on batch. Applies on elements one by one by default. 
        Should be implemented if more efficient batch application is possible.
        '''

        return [self.apply(elem) for elem in batch]

    def get_torch_model(self):
        '''
        Returns torch model if exists. Returns None by default.
        '''
        return None

    def train(self):
        '''
        Runs model training if needed.
        '''
        trainer = EmbedderTrainer(
            embedder=self.model,
            heads={
                
            }
        )

    def load_base_model(self):

        return model