#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

import cartpole

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

RANDOM_SEED = 42
EVAL_EPISODES = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax, mutable_layer = tf.keras.layers.Dense(9, kernel_initializer="glorot_normal"))
toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation, seed=RANDOM_SEED, episodes=EVAL_EPISODES)

def gen_pop(number):
    return [creator.Individual() for i in range(number)]
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cartpole.cartover)
toolbox.register("mutate", cartpole.mutcartion, sigma=1)
#toolbox.register("mutate", mutIdentity)


hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

final_pop = algorithms.eaMuPlusLambda(gen_pop(30),toolbox,30, 50, 0.3, 0.01, 10, stats, hof, True)
cartpole.evalutation(final_pop[0][0], RANDOM_SEED, 10, True)
