import numpy as np
import differential_evo
import tensorflow as tf
import statconf

from deap import algorithms
from deap import base
from deap import tools
class Replayble:
    def __init__(self):
        self.fit_attr = "fitness"
    def gen_pop(self):
        return [self.creator.Individual() for i in range(self.pop_size)]

    def replayBest(self):
        if self.final_pop is not None:
            best = tools.selBest(self.final_pop, 1,self.fit_attr)[0]
            self.replay_f(best)
        else:
            return False
        return True

class LambdaAlgContainer(Replayble):
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator):
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
        self.stats = statconf.get_statistics(lambda ind: ind.fitness.values)
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
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator):
        toolbox.register("map", self.novelty_operator)
        super().__init__(pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        self.stats = statconf.get_statistics(lambda ind: ind.fitness2.values)
        self.fit_attr = "fitness2"

    @staticmethod
    def novelty_operator(novelty_evaluation, evaluated_pop):
        fitness_novelty=[]
        behaviours = list(map(novelty_evaluation, evaluated_pop))
        for b, ind in zip(behaviours, evaluated_pop):
            fitness, beh = b
            ind.fitness2.values=(fitness,)
            beh_distances = map(lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), behaviours)
            novelty = np.mean(np.array(list(beh_distances)))
            fitness_novelty.append([novelty])
        return fitness_novelty

class DiffNoveltyContainer(DiffAlgContainer):
    def __init__(self, pop, toolbox, seed, ngen, creator, statistics):
        super().__init__(pop, toolbox, seed, ngen, creator, statistics)
        
    def run(self):
        tf.keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = differential_evo.differential_evolatuion(self.gen_pop(),self.toolbox, self.ngen, self.stats, self.hof, True)
        return self.final_pop, self.logbook 