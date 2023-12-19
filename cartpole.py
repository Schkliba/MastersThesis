import tensorflow as tf
import numpy as np
import gymnasium as gym
from typing import *
def evalutation_b(individual: tf.keras.Model, seed:int, episodes:int, replay=False) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
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
            if len(prediction) == 1:
                action = 1 if prediction[0] > 0.5 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
    return total_score / episodes, [observation[0]/4.8, observation[2]/0.418]

def evalutation(individual: tf.keras.Model, seed:int, episodes:int, replay=False) -> float:
    fit, b = evalutation_b(individual, seed, episodes, replay)
    return fit,

NEURON_COUNT = 4

class CartpolePlayer(tf.keras.Model):
    def __init__(self, mut_l=None):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(4)
        self.mutable_layer = tf.keras.layers.Dense(NEURON_COUNT, activation=tf.keras.activations.tanh)
        if mut_l is not None:
            self.mutable_layer.set_weights(mut_l)
        self.d_out = tf.keras.layers.Dense(2)

    def call(self, input):
        x = self.d1(input)
        x = self.mutable_layer(x)
        return self.d_out(x)
    

#Cross overs
def switcheroo(ind1, ind2):
    return ind2, ind1 

def cart_mean(ind1:CartpolePlayer, ind2:CartpolePlayer):
    u = ind1.mutable_layer.get_weights()
    v = ind2.mutable_layer.get_weights()
    f = [(u[i] + v[i])/2 for i in range(len(u)) ]
    ind1.mutable_layer.set_weights(f)
    ind2.mutable_layer.set_weights(f)
    return ind1, ind2

def cart_uniform(ind1:CartpolePlayer, ind2:CartpolePlayer, prob_filter):
    u = ind1.mutable_layer.get_weights()
    v = ind2.mutable_layer.get_weights()
    newind1 = []
    newind2 = []
    for i, mat in enumerate(u):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, v[i])
        newind1.append(n)
        n = np.where(cpoints, v[i], mat)
        newind2.append(n)
    ind1.mutable_layer.set_weights(newind1)
    ind2.mutable_layer.set_weights(newind2)
    return ind1, ind2


# Mutations
def mutIdentity(individual):
    return individual

def mutcartion(individual, sigma):
    ws = individual.mutable_layer.get_weights()
    rand = [w + np.random.normal(0, sigma, w.shape) for w in ws]
    individual.mutable_layer.set_weights(rand)
    return individual,

# Defferentials
def cartdiff(base:CartpolePlayer, diff1:CartpolePlayer, diff2:CartpolePlayer, alpha:float):
    a = base.mutable_layer.get_weights()
    b = diff1.mutable_layer.get_weights()
    c = diff2.mutable_layer.get_weights()
    n = [(a[i] + alpha * (b[i] - c[i])) for i in range(len(a))]
    return n

def cartrecomb(ind:CartpolePlayer, mats, prob_filter):
    newind = []
    og = ind.mutable_layer.get_weights()
    for i, mat in enumerate(mats):
        cpoints = np.random.random(mat.shape) < prob_filter
        n = np.where(cpoints, mat, og[i])
        newind.append(n)
    ind.mutable_layer.set_weights(newind)
    return ind

