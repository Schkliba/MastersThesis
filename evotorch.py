import torch

class CartpoleTorch:
    def __init__(self, mut_l=None):
        self._model = torch.nn.Sequential(
            torch.nn.Linear(4,4),
            torch.nn.Tanh(),
            torch.nn.Linear(4,2),
            torch.nn.Softmax()
        )

    def predict(self, inputs):
        return self._model(inputs)
    
    def get_agent_weights(self):
        return self.mutable_layer.get_weights()

    def set_agent_weights(self, weights):
        self.mutable_layer.set_weights(weights)