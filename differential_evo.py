import cartpole
import random
import numpy as np
from deap import base
from deap import tools

        
def mutation(pop, toolbox):
    go_out = []
    for j in range(len(pop)):
        candidates = [candidate for i,candidate in enumerate(pop) if i != j]
        a, b, c = tools.selRoulette(candidates, 3)
        new_ind = toolbox.triOp(a, b, c)
        trial = toolbox.recombine(toolbox.clone(pop[j]), new_ind)
        trial.fitness.values = toolbox.evaluate(trial)
        outputed = tools.selBest([trial, pop[j]], 1)[0] 
        go_out.append(outputed)
    return go_out


def differential_evolatuion(population, toolbox, ngen, stats, hof, verbose=True):
    logbook = tools.Logbook()
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    for gen in range(ngen):
        population = mutation(population, toolbox)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)

    return population, logbook