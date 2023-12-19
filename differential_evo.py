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


def novelty_mutation(pop, toolbox):

    new_ids = []
    for j in range(len(pop)):
        candidates = [candidate for i,candidate in enumerate(pop) if i != j]
        a, b, c = tools.selRandom(candidates, 3)
        new_ind = toolbox.triOp(a, b, c)
        new_ids.append(new_ind)
    trials = list(map(lambda x: toolbox.recombine(x[0], x[1]), zip(pop, new_ids)))
    fitnesses = toolbox.map(toolbox.evaluate, trials)
    for ind, fit in zip(trials, fitnesses):
        ind.fitness.values = fit

    outputed = map(lambda x: tools.selBest([x[1], x[0]], 1)[0], zip(pop, list(trials)))
    return list(outputed)
    """
    for j in range(len(pop)):
        candidates = [candidate for i,candidate in enumerate(pop) if i != j]
        a, b, c = tools.selRoulette(candidates, 3)
        new_ind = toolbox.triOp(a, b, c)
        trial = toolbox.recombine(toolbox.clone(pop[j]), new_ind)
        trial.fitness2.values,behavior = toolbox.evaluate(trial)
        outputed = tools.selBest([trial, pop[j]], 1)[0] 
        go_out.append(outputed)
    return go_out
    """
    

def differential_evolatuion(population, toolbox, ngen, stats, hof, verbose=True):
    logbook = tools.Logbook()

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(ngen):
        population = toolbox.mutation(toolbox.clone(population), toolbox)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        if verbose: print(logbook.stream)

    return population, logbook