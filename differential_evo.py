import cartpole
import numpy as np
from deap import base

        
def mutation(pop, toobox):
    trials = []
    for j in range(len(pop)):
        candidates = [candidate for candidate in range(pop_size) if candidate != j]
        a, b, c = pop[choice(candidates, 3, replace=False)]
        new_ind = toolbox.triOp(a, b, c)
        trial = toobox.remobination(pop[j], new_ind)
        trials.append(trial)
    return trials


def differential_evolatuion(population, toolbox, recombination_c, ngen, stats, hof, verbose=True):
    for gen in range(ngen):
        trials = mutation(population, toolbox)
        population = toolbox.select(pop, trials)
        record = stats.compile(population)
        if verbose: print(record)