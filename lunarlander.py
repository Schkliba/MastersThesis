import keras
import numpy as np
import gymnasium as gym
import libs.agent_infra as ai
from typing import *

class LunarLanderEvaluator(ai.Evaluator):

    hidden_dim = 20

    def __init__(self, replay=False, hidden_dim=None):
        super().__init__()
        self.enviroment = gym.make("LunarLander-v2", render_mode=None if not replay else "human")

        self.in_dim = self.enviroment.observation_space.shape
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        self.out_dim = self.enviroment.action_space.shape

    class LunarLanderAgent(ai.Player):
        def __init__(self, hidden_dim=7, mut_l=None):
            super().__init__()
            self.d1 = keras.layers.Dense(self.in_dim)
            self.mutable_layer = keras.layers.Dense(hidden_dim, activation="tanh")
            if mut_l is not None:
                self.mutable_layer.set_weights(mut_l)
            self.d_out = keras.layers.Dense(self.out_dim, activation="softmax")

    def get_individual_base(self):
        self.LunarLanderAgent.in_dim = (self.enviroment.observation_space.shape)[0]
        self.LunarLanderAgent.out_dim = self.enviroment.action_space.n
        return self.LunarLanderAgent

    def prepare_toolbox(self, toolbox, ind_f):
        toolbox.register(
            "gen_individual", 
            ind_f, 
            hidden_dim=self.hidden_dim
        )
        toolbox.register("gen_pop", self.gen_pop ,ind_f=toolbox.gen_individual)




