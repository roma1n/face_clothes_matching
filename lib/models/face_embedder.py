from deepface import DeepFace
from torch import nn

import config
from lib.models.abstract_model import AbstractModel


class FaceEmbedder(AbstractModel, nn.Module):
    def __init__(self):
        AbstractModel.__init__(self)
        nn.Module.__init__(self)

        self.model = self.load_base_model()

    def apply(self, input):
        return DeepFace.represent(input, model=self.model, detector_backend='ssd')

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
        pass

    def load_base_model(self):
        return DeepFace.build_model(config.DEEPFACE_MODEL)
