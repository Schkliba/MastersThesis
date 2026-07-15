import keras
import copy
import numpy as np
import os
import libs.agent_infra as ai
import gymnasium as gym
from typing import *

# Defining behavior and starting conditions of the Carpole task

class CartpoleEvaluator(ai.Evaluator):
    hidden_dim = 4
    enviroment = gym.make("CartPole-v1", render_mode=None)
    in_dim = (enviroment.observation_space.shape)[0]
    out_dim = enviroment.action_space.n
    def __new__(cls, replay=False, hidden_dim=None, behavioral_space_f=None):
        cls.enviroment = gym.make("CartPole-v1", render_mode=None)
        cls.in_dim = (cls.enviroment.observation_space.shape)[0]
        cls.out_dim = cls.enviroment.action_space.n
        return super(CartpoleEvaluator, cls).__new__(cls)


    def __init__(self, replay=False, hidden_dim=None, behavioral_space_f=None):
        super().__init__()
        self.enviroment = gym.make("CartPole-v1", render_mode=None if not replay else "human")
        if behavioral_space_f is None:
            self.behavior_space_f = lambda observation, b: [observation[0]/4.8, observation[2]/0.418]
        else:
            self.behavior_space_f = behavioral_space_f

    def get_individual_base(self):
        return CartpoleAgent

    def prepare_toolbox(self, toolbox, ind_f):
        toolbox.register(
            "gen_individual", 
            ind_f, 
            hidden_dim=self.hidden_dim
        )
        toolbox.register("gen_pop", self.gen_pop ,ind_f=toolbox.gen_individual)

class CartpoleAgent(ai.Player):

    def __init__(self, hidden_dim=4, mut_l=None, trainable=False, dtype="float32"):
        super().__init__(trainable=False, dtype=dtype)
        self.d1 = keras.layers.Dense(CartpoleEvaluator.in_dim)

        self.mutable_layer = keras.layers.Dense(hidden_dim, activation="tanh")
        if mut_l is not None:
            self.mutable_layer.set_weights(mut_l)
        
        self.d_out = keras.layers.Dense(CartpoleEvaluator.out_dim)

if __name__ == "__main__":
    eva = CartpoleEvaluator()
    ind = eva.get_individual_base()()
    print(ind.get_agent_weights())

