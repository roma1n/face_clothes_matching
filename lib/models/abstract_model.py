import abc


class AbstractModel:
    @abc.abstractmethod
    def apply(self, input, **kwargs):
        '''
        Model application on single object. Must be implemented specifically for each model.
        '''
        pass

    def apply_batched(self, batch, **kwargs):
        '''
        Model application on batch. Applies on elements one by one by default. 
        Should be implemented if more efficient batch application is possible.
        '''

        return [self.apply(elem, **kwargs) for elem in batch]

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
