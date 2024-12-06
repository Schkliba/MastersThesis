import numpy as np
import differential_evo as de
import keras
import statconf
import libs.archiving as archiving

from deap import algorithms
from deap import base
from deap import tools


class Replayble:
    """
    Mutex to ensure replay of the best individual
    """
    def __init__(self):
        self.fit_attr = "fitness"

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
    def novelty_operator(novelty_evaluation_f, evaluated_pop, eval_map=map):
        fitness_novelty=[]
        behaviours = list(eval_map(novelty_evaluation_f, evaluated_pop))
        for b, ind in zip(behaviours, evaluated_pop):
            fitness, beh = b
            ind.behaviour = beh
            ind.fitness2.values = fitness
            beh_distances = map( 
                lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                behaviours
            )
            novelty = np.mean(np.array(list(beh_distances)))
            fitness_novelty.append([novelty])
        return fitness_novelty

class AddedNovelty(PureNovelty):
    @staticmethod
    def addnovelty_operator(novelty_evaluation_f, evaluated_pop, eval_map=map, weight_n=1, weight_f=1):
        fitness_novelty=[]
        behaviours = list(eval_map(novelty_evaluation_f, evaluated_pop))
        for b, ind in zip(behaviours, evaluated_pop):
            fitness, beh = b
            ind.behaviour = beh
            ind.fitness2.values=fitness
            beh_distances = map(
                lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                behavioursfit_attr
            )
            novelty = np.mean(np.array(list(beh_distances)))
            fitness_novelty.append([weight_n * novelty + weight_f *fitness])
        return fitness_novelty

class LambdaAlgContainer(Replayble):
    """
    Implements classical evo strategy L+M
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, tournsize=3):
        super().__init__()
        self.pop_size = pop
        self.mun = offs
        self.mut_r = mut_rate
        self.cross_r = cross_rate
        self.seed = seed
        self.ngen = ngen
        self.toolbox = toolbox
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        self.hof = tools.HallOfFame(1)
        self.stats = statconf.get_statistics(lambda ind: ind.fitness.values)
        self.replay_f = self.replayBest
        self.creator = creator

    def run(self, verbose=True):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = algorithms.eaMuPlusLambda(
            self.toolbox.gen_pop(self.pop_size),
            self.toolbox, 
            self.pop_size, 
            self.mun,
            self.cross_r, 
            self.mut_r, 
            self.ngen, 
            self.stats, 
            self.hof, 
            verbose
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

       
    def run(self, verbose=True):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.differential_evolution(
            self.toolbox.gen_pop(self.pop_size),
            self.toolbox, 
            self.ngen, 
            self.stats, 
            self.hof, 
            verbose
        )
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

class DiffAdditionNoveltyContainer(DiffAlgContainer, AddedNovelty):
    """
    Implements differential evolution with additive mix of novelty and fitness
    """
    def __init__(self, pop, toolbox, seed, ngen, creator):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.addnovelty_operator)
        toolbox.register("mutation", de.novelty_mutation)

class DiffArchivingContainer(DiffAlgContainer):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.Archive(archiving_period, pop)

    def run(self,verbose=True):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.archiving_differential_evolution(
            self.toolbox.gen_pop(self.pop_size),
            self.toolbox, 
            self.ngen, 
            self.stats, 
            self.hof, 
            self.archive, 
            verbose
        )
        return self.final_pop, self.logbook 

class DiffArchivingNoveltyContainer(DiffAlgContainer, PureNovelty):
    """
    Implements diffential novelty evolution with basic archiving, pure novelty
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)
        toolbox.register("mutation", de.novelty_mutation)
        self.archive = archiving.NoveltyArchive(archiving_period, pop)

    def run(self, verbose=True):
        keras.utils.set_random_seed(self.seed)
        self.final_pop, self.logbook = de.archiving_differential_evolution(
            self.toolbox.gen_pop(self.pop_size),
            self.toolbox, 
            self.ngen, 
            self.stats, 
            self.hof, 
            self.archive, 
            verbose
        )
        return self.final_pop, self.logbook 

class LambdaArchivingContainer(LambdaAlgContainer):
    """
    Implements L+M strategy with basic archiving
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2):
        super().__init__(pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        self.archive = archiving.Archive(archiving_period, pop)
        toolbox.decorate("select", self.archiveOp)

    @property
    def archiveOp(self):
        def decorator(select):
            def wrapper(population, number, tournsize):
                population = select(population + self.archive.get_stored(), number, tournsize=tournsize)
                self.archive.store_individuals(population)
                return population
            return wrapper
        return decorator


class LambdaArchivingNoveltyContainer(LambdaArchivingContainer, PureNovelty):
    """
    Implements diffential evolution with basic archiving, pure novelty
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2):
        LambdaArchivingContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)        
        self.archive = archiving.NoveltyArchive(archiving_period, pop)
        toolbox.decorate("select", self.archiveOp)





