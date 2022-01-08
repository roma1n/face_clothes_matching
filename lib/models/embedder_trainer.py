import torch
from torch import nn

import config
from lib.models.abstract_model import AbstractModel


class EmbedderTrainer(AbstractModel, nn.Module):
    def __init__(self, embedder, heads_dict, embedding_size=config.EMBEDDING_SIZE):
        '''
        :param embedder: nn.Module, embedder to train
        :param heads: dict, head name (str) to head size (int)
        '''
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.embedder = embedder
        self.heads_dict = heads_dict
        self.embedding_size = embedding_size

        self.heads = {
            head_name: nn.Linear(self.embedding_size, self.heads_dict[head_name]) for head_name in self.heads_dict
        }

    def apply(self, x):
        with torch.no_grad():
            return self.forward(x)

    def forward(self, x):
        embedding = self.embedder(x)

        return {
            head_name: self.heads[head_name](embedding) for head_name in self.heads
        }
