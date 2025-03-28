import random
import libs.archiving as archiving
import numpy as np
from deap import base
from deap import tools

        
def mutation(pop, toolbox:base.Toolbox):
    go_out = []
    for j in range(len(pop)):
        candidates = [candidate for i,candidate in enumerate(pop) if i != j]
        #print(list(map(lambda x: x.fitness, candidates)))
        a, b, c = tools.selRandom(candidates, 3)
        new_ind = toolbox.triOp(a, b, c)
        cloned = toolbox.clone(pop[j])
        trial = toolbox.recombine(cloned, new_ind)
        trial.fitness.values = toolbox.evaluate(trial)
        outputed = tools.selBest([trial, pop[j]], 1)[0] 
        go_out.append(outputed)
    return go_out


def novelty_mutation(pop, toolbox:base.Toolbox):
    new_ids = []
    for j in range(len(pop)):
        candidates = [candidate for i,candidate in enumerate(pop) if i != j]
        a, b, c = tools.selRandom(candidates, 3)
        new_ind = toolbox.triOp(a, b, c)
        new_ids.append(new_ind)
    trials = list(map(lambda x: toolbox.recombine(x[0], x[1]), zip(pop, new_ids)))
    novelties = toolbox.map(toolbox.evaluate, trials) #writing in fitness2 and behaviour
    for ind, fit in zip(trials, novelties):
        ind.fitness.values = fit

    outputed = map(lambda x: tools.selBest([x[1], x[0]], 1)[0], zip(pop, list(trials)))
    return list(outputed)
    

def differential_evolution(population, toolbox, ngen, stats, hof, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(ngen):
        population = toolbox.mutation(population, toolbox)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        if verbose: print(logbook.stream)
        
    return population, logbook

def archiving_differential_evolution(population, toolbox, ngen, stats, hof, archive:archiving.Archive, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])
    pop_size = len(population)
    fitnesses = toolbox.map(toolbox.evaluate, population)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(ngen):
        population = tools.selBest(toolbox.mutation(population+archive.get_stored(), toolbox), pop_size)
        archive.store_individuals(population)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        if verbose: print(logbook.stream)

    return population, logbook