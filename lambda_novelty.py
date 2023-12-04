import containers as con
import cartpole
import numpy as np
from deap import base
from deap import creator

creator.register("FitnessNovelty", base.Fitness, weights=(1.0, 1.0))
creator.register("Behavior", base.Fitness, weights=(1.0, 1.0)) #end position & angle

creator.register("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty, behaviour=creator.Behavior)
