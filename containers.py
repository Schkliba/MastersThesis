import numpy as np
import differential_evo as de
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


class PureNovelty:

    def __init__(self):
        self.stats = statconf.get_novelty_stats(lambda ind: ind.fitness.values, lambda ind: ind.fitness2.values)
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

    def __init__(self, pop, toolbox, seed, ngen, creator):
        super().__init__()
        self.pop_size = pop
        self.seed = seed
        self.ngen = ngen
        self.toolbox = toolbox
        self.hof = tools.HallOfFame(1)
        self.stats = statconf.get_statistics(lambda ind: ind.fitness.values)
        self.replay_f = self.replayBest
        self.creator = creator
        toolbox.register("mutation", de.mutation)
       
    def run(self):
        tf.keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.differential_evolatuion(self.gen_pop(),self.toolbox, self.ngen, self.stats, self.hof, True)
        return self.final_pop, self.logbook 


class LambdaNoveltyAlg(LambdaAlgContainer, PureNovelty):

    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator):
        LambdaAlgContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)

        

class DiffNoveltyContainer(DiffAlgContainer, PureNovelty):

    def __init__(self, pop, toolbox, seed, ngen, creator):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)
        toolbox.register("mutation", de.novelty_mutation)


    