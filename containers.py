import numpy as np

from deap import algorithms
from deap import base
from deap import tools
class LambdaAlgContainer:
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, replay_f):
        self.lambdan = pop
        self.mun = offs
        self.mut_r = mut_rate
        self.cross_r = cross_rate
        self.seed = seed
        self.ngen = ngen
        self.toolbox = toolbox
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.replay_f = replay_f
        self.creator = creator
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def gen_pop(self):
        return [self.creator.Individual() for i in range(self.lambdan)]

    def run(self):
        self.final_pop, self.logbook = algorithms.eaMuPlusLambda(self.gen_pop(),self.toolbox, self.lambdan, self.mun,
                                                     self.cross_r, self.mut_r, self.ngen, self.stats, self.hof, True)

    def replayBest(self):
        if self.final_pop is not None:
            best = tools.selBest(self.final_pop, 1)[0]
            self.replay_f(best)
        else:
            return False
        return True