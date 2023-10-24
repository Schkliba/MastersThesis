#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gymnasium as gym

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


def cartpole_evalutation(individual: tf.keras.Model, seed:int, episodes:int) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    # Create the environment
    env = gym.make("CartPole-v1", render_mode=None)
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
    return total_score / episodes, 

class CartpolePlayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(4)
        self.d_out = tf.keras.layers.Dense(2)

    def call(self, input):
        x = self.d1(input)
        x = self.mutable_layer(x)
        return self.d_out(x)

def cartover(ind1, ind2):
    return ind2, ind1 #TODO: actual crossover 

def mutIdentity(individual):
    return individual

def mutcartion(individual, sigma):
    ws = individual.mutable_layer.get_weights()
    rand = [w + np.random.normal(0, sigma, w.shape) for w in ws]
    individual.mutable_layer.set_weights(rand)
    return individual,



RANDOM_SEED = 42
EVAL_EPISODES = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", CartpolePlayer,fitness=creator.FitnessMax,mutable_layer = tf.keras.layers.Dense(9, kernel_initializer="glorot_normal"))
toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole_evalutation, seed=RANDOM_SEED, episodes=EVAL_EPISODES)

def gen_pop(number):
    return [creator.Individual() for i in range(number)]
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cartover)
toolbox.register("mutate", mutcartion, sigma=1)
#toolbox.register("mutate", mutIdentity)


hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

final_pop = algorithms.eaMuCommaLambda(gen_pop(3),toolbox,3, 5, 0.3, 0.01, 10, stats, hof, True)
