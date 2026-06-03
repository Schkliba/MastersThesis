import numpy as np
import differential_evo as de
import keras
import statconf
import libs.archiving as archiving
import evolutionary_strategy as es

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
    def __init__(self, ngen):
        super().__init__()
        self.gen_counter = 0
        self.ngen = ngen
    @property
    def add_novelty_operator(self):
        def actual_operator(novelty_evaluation_f, evaluated_pop, eval_map=map, start_fit_w = 0.2, decay=2):
            max_novelty = None
            max_fitness = None
            min_novelty = None
            min_fitness = None
            fitness_novelty = []
            behaviours = list(eval_map(novelty_evaluation_f, evaluated_pop))
            for b, ind in zip(behaviours, evaluated_pop):

                fitness, beh = b
                ind.behaviour = beh
                ind.fitness2.values=fitness
                beh_distances = map(
                    lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                    behaviours
                )
                novelty = np.mean(np.array(list(beh_distances)))
                if min_novelty is None:
                    min_novelty=novelty
                else:
                    min_novelty=min(novelty,min_novelty)
                if max_novelty is None:
                    max_novelty=novelty
                else:
                    max_novelty = max(max_novelty, novelty)
                if min_fitness is None:
                    min_fitness=fitness[0]
                else:
                    min_fitness=min(fitness[0],min_fitness)
                if max_fitness is None:
                    max_fitness=fitness[0]
                else:
                    max_fitness = max(fitness[0], max_fitness)
                ind.fitness3 = novelty

            
            for ind in evaluated_pop:
                novelty = ind.fitness3
                fitness = ind.fitness2.values[0]
                scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                t = self.gen_counter/self.ngen
                W = start_fit_w * np.exp(-decay * t)
                #W = start_fit_w * (1-(self.gen_counter/self.ngen)) + (self.gen_counter/self.ngen) 
                fitness_novelty.append([W * scaled_novelty + (1-W) * scaled_fitness])

            return fitness_novelty
        return actual_operator

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
        self.final_pop, self.logbook = es.eaMuPlusLambda(
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

class LambdaAddNoveltyContainer(LambdaAlgContainer, AddedNovelty):
    """
    Implements evolutionary strategy L+M with mixed novelty
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator,fit_w, decay):
        LambdaAlgContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        AddedNovelty.__init__(self, ngen)
        toolbox.register("map", self.add_novelty_operator, start_fit_w=fit_w, decay=decay)     

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
    def __init__(self, pop, toolbox, seed, ngen, creator, fit_w, decay):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        AddedNovelty.__init__(self, ngen)
        toolbox.register("map", self.add_novelty_operator, start_fit_w=fit_w, decay=decay)
        toolbox.register("mutation", de.novelty_mutation)

class DiffArchivingContainer(DiffAlgContainer):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2, store_batch=1):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.Archive(archiving_period, pop, store_batch=store_batch)

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
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2, store_batch=1):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)
        toolbox.register("mutation", de.novelty_mutation)
        self.archive = archiving.Archive(archiving_period, pop, store_batch=store_batch)

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
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2, store_batch=1):
        super().__init__(pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        self.archive = archiving.Archive(archiving_period, pop, store_batch=store_batch)
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
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2, store_batch=1):
        LambdaArchivingContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)        
        self.archive = archiving.Archive(archiving_period, pop, store_batch)
        toolbox.decorate("select", self.archiveOp)





