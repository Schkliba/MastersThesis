import keras
import numpy as np
import gymnasium as gym
import libs.agent_infra as ai
from typing import *

class LunarLanderBehaviorModel:
    norm_cnst = {
            "x": 2.5,
            "y": 2.5,
            "x_v": 10,
            "y_v": 10,
            "angular_v": 10,
            "angle": 6.2831855
    }
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_x_v = 0

    def inter_call(self,observation):
        pass
    
    def transform(self,behavior, n):
        pass

        
class LunarLanderEvaluator(ai.Evaluator):

    hidden_dim = 8

    def __new__(cls, replay=False, hidden_dim=None, behavioral_space_f=None):
        cls.enviroment = gym.make("LunarLander-v2", render_mode=None if not replay else "human")
        cls.in_dim = (cls.enviroment.observation_space.shape)[0]
        cls.out_dim = cls.enviroment.action_space.n
        return super(LunarLanderEvaluator, cls).__new__(cls)
    
    def __init__(self, replay=False, hidden_dim=None, behavioral_space_f=None):
        super().__init__()
        self.enviroment = gym.make("LunarLander-v2", render_mode=None if not replay else "human")
        if behavioral_space_f is None:
            self.behavior_space_f = lambda observation: [observation[0]/2.5, observation[4]/6.2831855]
        else:
            self.behavior_space_f = behavioral_space_f
        #self.in_dim = self.enviroment.observation_space.shape
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        #self.out_dim = self.enviroment.action_space.shape

    
    class LunarLanderAgent:
        def __init__(self, hidden_dim=7, mut_l=None):
            super().__init__()
            self.in_dim = LunarLanderEvaluator.in_dim
            self.out_dim = LunarLanderEvaluator.out_dim
            self.d1 = keras.layers.Dense(self.in_dim)
            self.mutable_layer1 = keras.layers.Dense(hidden_dim, activation="tanh")
            self.mutable_layer2 = keras.layers.Dense(hidden_dim, activation="tanh")
            if mut_l is not None:
                self.mutable_layer1.set_weights(mut_l)
                self.mutable_layer2.set_weights(mut_l)
            self.d_out = keras.layers.Dense(self.out_dim, activation="softmax")
        
        def get_agent_weights(self):
            m1 = self.mutable_layer1.get_weights()
            m2 = self.mutable_layer2.get_weights()
            return m1 + m2

        def set_agent_weights(self, weights):
            mstandard = self.mutable_layer1.get_weights()
            m1 = weights[:len(mstandard)]
            m2 = weights[len(mstandard):]
            self.mutable_layer1.set_weights(m1)
            self.mutable_layer2.set_weights(m2)

        def __call__(self, inputs):
            x = self.d1(inputs)
            x = self.mutable_layer1(x)
            x = self.mutable_layer2(x)
            return self.d_out(x)

    def get_individual_base(self):
        
        return self.LunarLanderAgent

    def prepare_toolbox(self, toolbox, ind_f):
        toolbox.register(
            "gen_individual", 
            ind_f, 
            hidden_dim=self.hidden_dim
        )
        toolbox.register("gen_pop", self.gen_pop ,ind_f=toolbox.gen_individual)


if __name__ == "__main__":
    eva = LunarLanderEvaluator()
    ind = eva.get_individual_base()(hidden_dim=5)
    print(eva.enviroment.observation_space.shape)
    inputs = np.array([8]*8)[np.newaxis]
    ind(inputs=inputs)
    print(len(ind.get_agent_weights()[:2]))
    

