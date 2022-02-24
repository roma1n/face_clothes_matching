from deepface import DeepFace
from torch import nn

import config
from lib.models.abstract_model import AbstractModel


class FaceEmbedder(AbstractModel):
    def __init__(self, deepface_model=None):
        self.deepface_model = deepface_model or config.DEEPFACE_MODEL
        AbstractModel.__init__(self)

        self.model = self.load_base_model()

    def apply(self, input):
        return DeepFace.represent(input, model=self.model, detector_backend='ssd')

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
        return DeepFace.build_model(self.deepface_model)
