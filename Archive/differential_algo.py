#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

import cartpole
import seaborn as sns
import visualisation as vis
import differential_evo as de

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

RANDOM_SEED = 42
EVAL_EPISODES = 10

tf.keras.utils.set_random_seed(RANDOM_SEED)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax, mutable_layer = tf.keras.layers.Dense(9, kernel_initializer="glorot_normal"))
toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation, seed=RANDOM_SEED, episodes=EVAL_EPISODES)

def gen_pop(number):
    return [creator.Individual() for i in range(number)]
toolbox.register("triOp", cartpole.cartdiff, alpha=0.9)
toolbox.register("recombine", cartpole.cartrecomb, prob_filter=0.6)
toolbox.register("mutation", de.mutation)

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

final_pop, logbook = de.differential_evolatuion(gen_pop(5),toolbox, 10, stats, hof, True)
stats = ["avg", "std", "min", "max"]
df = vis.logbook2pandas(logbook, stats)
sns.lineplot(df["avg"])
#cartpole.evalutation((tools.selBest(final_pop, 1)[0]), RANDOM_SEED, 2, True)