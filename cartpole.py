import keras
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "torch"
import gymnasium as gym
from typing import *
def evalutation_b(individual: keras.Model, seed:int, episodes:int, replay=False) -> float:
    """
    Returns the average score achieved on the given number of episodes and normalised behaviour.
    """
    # Create the environment
    env = gym.make("CartPole-v1", render_mode=None if not replay else "human")
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

NEURON_COUNT = 4

class CartpolePlayer(keras.Model):
    def __init__(self, mut_l=None, trainable=False, dtype="float32"):
        super().__init__(trainable=False, dtype=dtype)
        self.d1 = keras.layers.Dense(4)
        self.mutable_layer = keras.layers.Dense(NEURON_COUNT, activation="tanh")
        if mut_l is not None:
            self.mutable_layer.set_weights(mut_l)
        self.d_out = keras.layers.Dense(2)

    def call(self, input):
        x = self.d1(input)
        x = self.mutable_layer(x)
        return self.d_out(x)
    
    def get_agent_weights(self):
        return self.mutable_layer.get_weights()

    def set_agent_weights(self, weights):
        self.mutable_layer.set_weights(weights)



#Cross overs
def switcheroo(ind1, ind2):
    return ind2, ind1 

def mean_cross(ind1:CartpolePlayer, ind2:CartpolePlayer):
    u = ind1.get_agent_weights()
    v = ind2.get_agent_weights()
    f = [(u[i] + v[i])/2 for i in range(len(u)) ]
    ind1.set_agent_weights(f)
    ind2.set_agent_weights(f)
    return ind1, ind2

def uniform_cross(ind1:CartpolePlayer, ind2:CartpolePlayer, prob_filter):
    u = ind1.get_agent_weights()
    v = ind2.get_agent_weights()
    newind1 = []
    newind2 = []
    for i, mat in enumerate(u):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, v[i])
        newind1.append(n)
        n = np.where(cpoints, v[i], mat)
        newind2.append(n)
    ind1.set_agent_weights(newind1)
    ind2.set_agent_weights(newind2)
    return ind1, ind2


# Mutations
def mutIdentity(individual):
    return individual

def mutation_func(individual, sigma):
    ws = individual.get_agent_weights()
    rand = [w + np.random.normal(0, sigma, w.shape) for w in ws]
    individual.set_agent_weights(rand)
    return individual,

# Defferentials
def tri_op(base:CartpolePlayer, diff1:CartpolePlayer, diff2:CartpolePlayer, alpha:float):
    a = base.get_agent_weights()
    b = diff1.get_agent_weights()
    c = diff2.get_agent_weights()
    n = [(a[i] + alpha * (b[i] - c[i])) for i in range(len(a))]
    return n

def recombine(ind:CartpolePlayer, mats, prob_filter):
    newind = []
    og = ind.get_agent_weights()
    for i, mat in enumerate(mats):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, og[i])
        newind.append(n)
    ind.set_agent_weights(newind)
    return ind

