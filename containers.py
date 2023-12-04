import numpy as np
import differential_evo
import tensorflow as tf

from deap import algorithms
from deap import base
from deap import tools
class Replayble:
    def gen_pop(self):
        return [self.creator.Individual() for i in range(self.pop_size)]

    def replayBest(self):
        if self.final_pop is not None:
            best = tools.selBest(self.final_pop, 1)[0]
            self.replay_f(best)
        else:
            return False
        return True

class LambdaAlgContainer(Replayble):
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, statistics):
        super().__init__()
        self.pop_size = pop
        self.mun = offs
        self.mut_r = mut_rate
        self.cross_r = cross_rate
        self.seed = seed
        self.ngen = ngen
        self.toolbox = toolbox
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.hof = tools.HallOfFame(1)
        self.stats = statistics
        self.replay_f = self.replayBest
        self.creator = creator

    def run(self):
        tf.keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = algorithms.eaMuPlusLambda(self.gen_pop(),self.toolbox, self.pop_size, self.mun,
                                                     self.cross_r, self.mut_r, self.ngen, self.stats, self.hof, True)
        return self.final_pop, self.logbook 

    

class DiffAlgContainer(Replayble):
    def __init__(self, pop, toolbox, seed, ngen, creator, statistics):
        super().__init__()
        self.pop_size = pop
        self.seed = seed
        self.ngen = ngen
        self.toolbox = toolbox
        self.hof = tools.HallOfFame(1)
        self.stats = statistics
        self.replay_f = self.replayBest
        self.creator = creator
        
    def run(self):
        tf.keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = differential_evo.differential_evolatuion(self.gen_pop(),self.toolbox, self.ngen, self.stats, self.hof, True)
        return self.final_pop, self.logbook 

class LambdaNoveltyAlg(LambdaAlgContainer):
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, replay_f):
        
        toolbox.register("VarOr", self.novelty_operator)
        super().__init__(pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, replay_f) 

    @staticmethod
    def novelty_operator():
        pass