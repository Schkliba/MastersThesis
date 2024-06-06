import numpy as np
import differential_evo as de
import keras
import statconf
import archiving

from deap import algorithms
from deap import base
from deap import tools


class Replayble:
    """
    Mutex to ensure replay of the best individual
    """
    def __init__(self):
        self.fit_attr = "fitness"

    def gen_pop(self):
        return [self.creator.Individual() for i in range(self.pop_size)]

    def replayBest(self):
        if self.final_pop is not None:
            best = tools.selBest(self.final_pop, 1, self.fit_attr)[0]
            self.replay_f(best)
            return True
        else:
            return False


class PureNovelty:
    """
    Implements conversion of behavior (second output of evaluation) 
    to novelty
    """
    def __init__(self):
        self.stats = statconf.get_novelty_stats(
            lambda ind: ind.fitness.values, 
            lambda ind: ind.fitness2.values
        )
        self.fit_attr = "fitness2"

    @staticmethod
    def novelty_operator(novelty_evaluation, evaluated_pop):
        fitness_novelty=[]
        behaviours = list(map(novelty_evaluation, evaluated_pop))
        for b, ind in zip(behaviours, evaluated_pop):
            fitness, beh = b
            ind.behaviour = beh
            ind.fitness2.values=(fitness,)
            beh_distances = map(
                lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                behaviours
            )
            novelty = np.mean(np.array(list(beh_distances)))
            fitness_novelty.append([novelty])
        return fitness_novelty


class LambdaAlgContainer(Replayble):
    """
    Implements classical evo strategy L+M
    """
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
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = algorithms.eaMuPlusLambda(
            self.gen_pop(),self.toolbox, self.pop_size, self.mun,
            self.cross_r, self.mut_r, self.ngen, self.stats, self.hof, True
        )
        return self.final_pop, self.logbook 


class DiffAlgContainer(Replayble):
    """
    Implements classical differential evolution
    """
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
        self.toolbox.register("mutation", de.mutation)

       
    def run(self):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.differential_evolution(self.gen_pop(),self.toolbox, self.ngen, self.stats, self.hof, True)
        return self.final_pop, self.logbook 


class LambdaNoveltyAlg(LambdaAlgContainer, PureNovelty):
    """
    Implements evolutionary strategy L+M with pure novelty
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator):
        LambdaAlgContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)        

class DiffNoveltyContainer(DiffAlgContainer, PureNovelty):
    """
    Implements differential evolution with pure novelty
    """
    def __init__(self, pop, toolbox, seed, ngen, creator):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)
        toolbox.register("mutation", de.novelty_mutation)

class DiffArchivingContainer(DiffAlgContainer):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.Archive(archiving_period, pop)

    def run(self):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.archiving_differential_evolution(self.gen_pop(),\
            self.toolbox, self.ngen, self.stats, self.hof, self.archive, True)
        return self.final_pop, self.logbook 

class DiffArchivingNoveltyContainer(DiffAlgContainer, PureNovelty):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.NoveltyArchive(archiving_period, pop)

    def run(self):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.archiving_differential_evolution(self.gen_pop(),\
            self.toolbox, self.ngen, self.stats, self.hof, self.archive, True)
        return self.final_pop, self.logbook 

class LambdaArchivingContainer(LambdaAlgContainer):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.Archive(archiving_period, pop)

    def archiveOp():
        def decorator(varOr):
            def wrapper(population, toolbox, lambda_, cxpb, mutpb):
                
                offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        return decorator


    