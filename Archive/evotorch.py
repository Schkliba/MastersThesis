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
        return self._model.parameters()

    def set_agent_weights(self, weights):
        for i, w in enum(weights):
            self._model[i].weights = w