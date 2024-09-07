import keras
import copy
import numpy as np
import os
import libs.agent_infra as ai
import gymnasium as gym
from typing import *

class CartpoleEvaluator(ai.Evaluator):
    hidden_dim = 4

    def __init__(self, replay=False, hidden_dim=None):
        super().__init__()
        self.enviroment = gym.make("CartPole-v1", render_mode=None if not replay else "human")
  


    class CartpoleAgent(ai.Player):
        def __deepcopy__(self, memo):
            new = self.__class__(self)
            new.mutable_layer = copy.deepcopy(self.mutable_layer, memo)
            return new

        def __init__(self, hidden_dim=4, mut_l=None, trainable=False, dtype="float32"):
            super().__init__(trainable=False, dtype=dtype)
            self.d1 = keras.layers.Dense(self.in_dim)

            self.mutable_layer = keras.layers.Dense(hidden_dim, activation="tanh")
            if mut_l is not None:
                self.mutable_layer.set_weights(mut_l)
          
            self.d_out = keras.layers.Dense(self.out_dim)


    def get_individual_base(self):
        self.CartpoleAgent.in_dim = (self.enviroment.observation_space.shape)[0]
        self.CartpoleAgent.out_dim = self.enviroment.action_space.n
        return self.CartpoleAgent

    def prepare_toolbox(self, toolbox, ind_f):
        toolbox.register(
            "gen_individual", 
            ind_f, 
            hidden_dim=self.hidden_dim
        )
        toolbox.register("gen_pop", self.gen_pop ,ind_f=toolbox.gen_individual)


