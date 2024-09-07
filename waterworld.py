import keras
import numpy as np
import gymnasium as gym
from pettingzoo.sisl import waterworld_v4

from typing import *

NEURON_COUNT = 4

class WaterworldEvaluator:
    def evalutation_b(individual: keras.Model, seed:int, episodes:int, replay=False) -> float:
        """
        Returns the average score achieved on the given number of episodes and normalised behaviour.
        """
        # Create the environment
        env = gym.make("LunarLander-v2", render_mode=None if not replay else "human")
        env.reset(seed=seed)

        # Evaluate the episodes
        total_score = 0
        for episode in range(episodes):
            observation, score, done = env.reset()[0], 0, False
            while not done:
                prediction = individual(observation[np.newaxis])[0].numpy()
                action = np.argmax(prediction)

                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                done = terminated or truncated

            total_score += score
        return total_score / episodes, [observation[0]/4.8, observation[2]/0.418]

    def evalutation(individual: keras.Model, seed:int, episodes:int, replay=False) -> float:
        fit, b = evalutation_b(individual, seed, episodes, replay)
        return fit,
    class LunarLanderAgent(keras.Model):
        def __init__(self, mut_l=None):
            super().__init__()
            self.d1 = keras.layers.Dense(4)
            self.mutable_layer = keras.layers.Dense(NEURON_COUNT, activation="tanh")
            if mut_l is not None:
                self.mutable_layer.set_weights(mut_l)
            self.d_out = keras.layers.Dense(2)

        def call(self, input):
            x = self.d1(input)
            x = self.mutable_layer(x)
            return self.d_out(x)


    

