import numpy as np
import keras
import libs.statconf as statconf
import libs.archiving as archiving
from libs import evolutionary_strategy as es, differential_evo as de

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
        self.max_novelty = None
        self.max_fitness = None
        self.min_novelty = None
        self.min_fitness = None
        self.last_behaviours = None
    
    def lambda_add_decorator(self, start_fit_w, decay):
        def actual_decorator(varOr):
            def new_var_Or(population, toolbox, lambda_, cxpb, mutpb):
                behaviours = list(map(lambda x: x.behaviour, population))
                self.last_behaviours = behaviours
                max_novelty = None
                max_fitness = None
                min_novelty = None
                min_fitness = None
                t = self.gen_counter/self.ngen
                W = start_fit_w * np.exp(-decay * t)
                for b, ind in zip(behaviours, population):
                    beh = b
                    fitness = ind.fitness2.values[0]
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
                        min_fitness=fitness
                    else:
                        min_fitness=min(fitness,min_fitness)
                    if max_fitness is None:
                        max_fitness=fitness
                    else:
                        max_fitness = max(fitness, max_fitness)
                    ind.fitness3 = novelty
                for ind in population:
                    novelty = ind.fitness3
                    fitness = ind.fitness2.values[0]
                    scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                    scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                    ind.fitness.values = [W * scaled_novelty + (1-W) * scaled_fitness]
                self.max_fitness = max_fitness
                self.min_fitness = min_fitness
                self.max_novelty = max_novelty
                self.min_novelty = min_novelty
                
                return varOr(population, toolbox, lambda_, cxpb, mutpb)
            return new_var_Or
        return actual_decorator
    @property
    def dry_add_novelty_operator(self):
        def actual_operator( 
                pop, 
                start_fit_w = 0.2, 
                decay=2
            ):
            max_novelty = None
            max_fitness = None
            min_novelty = None
            min_fitness = None
            fitness_novelty = []
            behaviours = list(map(lambda x: x.behaviour, pop))
            self.last_behaviours = behaviours

            t = self.gen_counter/self.ngen
            W = start_fit_w * np.exp(-decay * t)
            for beh, ind in zip(behaviours, pop):

                beh_distances = map(
                    lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                    behaviours
                )
                fitness = ind.fitness2.values[0]
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
                    min_fitness=fitness
                else:
                    min_fitness=min(fitness,min_fitness)
                if max_fitness is None:
                    max_fitness=fitness
                else:
                    max_fitness = max(fitness, max_fitness)
                ind.fitness3 = novelty
            for ind in pop:
                novelty = ind.fitness3
                fitness = ind.fitness2.values[0]
                scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                
                #W = start_fit_w * (1-(self.gen_counter/self.ngen)) + (self.gen_counter/self.ngen) 
                fitness_novelty.append([W * scaled_novelty + (1-W) * scaled_fitness])
            self.max_fitness = max_fitness
            self.min_fitness = min_fitness
            self.max_novelty = max_novelty
            self.min_novelty = min_novelty
            return fitness_novelty
        return actual_operator
    @property
    def add_novelty_operator(self):
        def actual_operator(
                novelty_evaluation_f, 
                evaluated_pop,
                eval_map=map, 
                start_fit_w = 0.2, 
                decay=2
            ):
            max_novelty = self.max_novelty
            max_fitness = self.max_fitness
            min_novelty = self.min_novelty
            min_fitness = self.min_fitness
            fitness_novelty = []
            t = self.gen_counter/self.ngen
            W = start_fit_w * np.exp(-decay * t)
            print(f"Weight is now {W} - decay{decay}")

            behaviours = list(eval_map(novelty_evaluation_f, evaluated_pop))
            for b, ind in zip(behaviours, evaluated_pop):

                fitness, beh = b
                ind.behaviour = beh
                ind.fitness2.values=fitness
                beh_distances = map(
                    lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                    behaviours if self.last_behaviours is None else self.last_behaviours
                )
                novelty = np.mean(np.array(list(beh_distances)))
                if min_novelty is None:
                    min_novelty=novelty
                elif self.min_novelty is None:
                    min_novelty=min(novelty,min_novelty)
                if max_novelty is None:
                    max_novelty=novelty
                elif self.max_novelty is None:
                    max_novelty = max(max_novelty, novelty)
                if min_fitness is None:
                    min_fitness=fitness[0]
                elif self.min_fitness is None:
                    min_fitness=min(fitness[0],min_fitness)
                if max_fitness is None:
                    max_fitness=fitness[0]
                elif self.max_fitness is None:
                    max_fitness = max(fitness[0], max_fitness)
                ind.fitness3 = novelty
            
            print(f"Weight is now {W} - decay{decay}")

            for ind in evaluated_pop:
                novelty = ind.fitness3
                fitness = ind.fitness2.values[0]
                scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                
                #W = start_fit_w * (1-(self.gen_counter/self.ngen)) + (self.gen_counter/self.ngen) 
                fitness_novelty.append([W * scaled_novelty + (1-W) * scaled_fitness])
            self.gen_counter +=1
            return fitness_novelty
        return actual_operator
    
class SubNovelty(PureNovelty):
    def __init__(self, ngen):
        super().__init__()
        self.gen_counter = 0
        self.ngen = ngen
        self.max_novelty = None
        self.max_fitness = None
        self.min_novelty = None
        self.min_fitness = None
        self.last_behaviours = None 
    def lambda_sub_decorator(self, start_fit_w, decay):
        def actual_decorator(varOr):
            def new_var_Or(population, toolbox, lambda_, cxpb, mutpb):
                behaviours = list(map(lambda x: x.behaviour, population))
                self.last_behaviours = behaviours
                max_novelty = None
                max_fitness = None
                min_novelty = None
                min_fitness = None
                t = self.gen_counter/self.ngen
                W = start_fit_w * np.exp(-decay * t)
                for b, ind in zip(behaviours, population):
                    beh = b
                    fitness = ind.fitness2.values[0]
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
                        min_fitness=fitness
                    else:
                        min_fitness=min(fitness,min_fitness)
                    if max_fitness is None:
                        max_fitness=fitness
                    else:
                        max_fitness = max(fitness, max_fitness)
                    ind.fitness3 = novelty
                for ind in population:
                    novelty = ind.fitness3
                    fitness = ind.fitness2.values[0]
                    scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                    scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                    ind.fitness.values = [(1-W) * scaled_novelty + (W) * scaled_fitness]
                self.max_fitness = max_fitness
                self.min_fitness = min_fitness
                self.max_novelty = max_novelty
                self.min_novelty = min_novelty
                
                return varOr(population, toolbox, lambda_, cxpb, mutpb)
            return new_var_Or
        return actual_decorator
    @property
    def dry_sub_novelty_operator(self):
        def actual_operator( 
                pop, 
                start_fit_w = 0.2, 
                decay=2
            ):
            max_novelty = None#novelty_limits.max
            max_fitness = None#fitness_limits.max
            min_novelty = None#novelty_limits.min
            min_fitness = None#fitness_limits.min
            fitness_novelty = []
            behaviours = list(map(lambda x: x.behaviour, pop))
            self.last_behaviours = behaviours
            t = self.gen_counter/self.ngen
            W = start_fit_w * np.exp(-decay * t)

            for beh, ind in zip(behaviours, pop):
                fitness = ind.fitness2.values[0]

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
                    min_fitness=fitness
                else:
                    min_fitness=min(fitness,min_fitness)
                if max_fitness is None:
                    max_fitness=fitness
                else:
                    max_fitness = max(fitness, max_fitness)
                ind.fitness3 = novelty
            for ind in pop:
                novelty = ind.fitness3
                fitness = ind.fitness2.values[0]
                scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                
                #W = start_fit_w * (1-(self.gen_counter/self.ngen)) + (self.gen_counter/self.ngen) 
                fitness_novelty.append([(1-W) * scaled_novelty + (W) * scaled_fitness])
            self.max_fitness = max_fitness
            self.min_fitness = min_fitness
            self.max_novelty = max_novelty
            self.min_novelty = min_novelty
            return fitness_novelty
        return actual_operator
    @property
    def sub_novelty_operator(self):
        def actual_operator(
                novelty_evaluation_f, 
                evaluated_pop, 
                eval_map=map,
                start_fit_w = 0.2, 
                decay=2
            ):
            max_novelty = self.max_novelty
            max_fitness = self.max_fitness
            min_novelty = self.min_novelty
            min_fitness = self.min_fitness
            fitness_novelty = []
            behaviours = list(eval_map(novelty_evaluation_f, evaluated_pop))

            t = self.gen_counter/self.ngen
            W = start_fit_w * np.exp(-decay * t)
            print(f"Weight is now {W} - decay{decay} - gen{self.gen_counter}")
            for b, ind in zip(behaviours, evaluated_pop):

                fitness, beh = b
                ind.behaviour = beh
                ind.fitness2.values=fitness
                beh_distances = map(
                    lambda x: np.linalg.norm(np.array(x[1])-np.array(beh)), 
                    behaviours if self.last_behaviours is None else self.last_behaviours
                )
                novelty = np.mean(np.array(list(beh_distances)))
                if min_novelty is None:
                    min_novelty=novelty
                elif self.min_novelty is None:
                    min_novelty=min(novelty,min_novelty)
                if max_novelty is None:
                    max_novelty=novelty
                elif self.max_novelty is None:
                    max_novelty = max(max_novelty, novelty)
                if min_fitness is None:
                    min_fitness=fitness[0]
                elif self.min_fitness is None:
                    min_fitness=min(fitness[0],min_fitness)
                if max_fitness is None:
                    max_fitness=fitness[0]
                elif self.max_fitness is None:
                    max_fitness = max(fitness[0], max_fitness)
                ind.fitness3 = novelty

            
            for ind in evaluated_pop:
                novelty = ind.fitness3
                fitness = ind.fitness2.values[0]
                scaled_novelty =  0 if max_novelty == min_novelty else(novelty - min_novelty)/(max_novelty-min_novelty)
                scaled_fitness =  0 if max_fitness == min_fitness else (fitness - min_fitness)/(max_fitness-min_fitness)
                
                #W = start_fit_w * (1-(self.gen_counter/self.ngen)) + (self.gen_counter/self.ngen) 
                fitness_novelty.append([(1-W) * scaled_novelty + (W) * scaled_fitness])
            self.gen_counter +=1
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
        self.toolbox.register("varOver", es.varOver)

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
    def __init__(
            self, 
            pop, 
            offs, 
            mut_rate, 
            cross_rate, 
            seed, 
            ngen, 
            toolbox, 
            creator,
            fit_w, 
            decay, 
            fitness_limits, 
            novelty_limits
        ):
        LambdaAlgContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        AddedNovelty.__init__(self, ngen)
        toolbox.decorate("varOver", self.lambda_add_decorator(start_fit_w=fit_w, decay=decay))

        toolbox.register(
            "map", 
            self.add_novelty_operator, 
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits,
            start_fit_w=fit_w, 
            decay=decay
        )   

class LambdaSubNoveltyContainer(LambdaAlgContainer, SubNovelty):
    """
    Implements evolutionary strategy L+M with mixed novelty
    """
    def __init__(
            self, 
            pop, 
            offs, 
            mut_rate, 
            cross_rate, 
            seed, 
            ngen, 
            toolbox, 
            creator,
            fit_w, 
            decay,
            fitness_limits, 
            novelty_limits
        ):
        LambdaAlgContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        SubNovelty.__init__(self, ngen)
        toolbox.decorate("varOver", self.lambda_sub_decorator(start_fit_w=fit_w, decay=decay),)
        toolbox.register(
            "map", 
            self.sub_novelty_operator,
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits, 
            start_fit_w=fit_w, 
            decay=decay
        )       

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
    def __init__(
            self, 
            pop, 
            toolbox, 
            seed, 
            ngen, 
            creator, 
            fit_w, 
            decay,
            fitness_limits, 
            novelty_limits
        ):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        AddedNovelty.__init__(self, ngen)
        toolbox.register(
            "map", 
            self.add_novelty_operator, 
            start_fit_w=fit_w, 
            decay=decay,
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits 
        )
        toolbox.register(
            "dry_map", 
            self.dry_add_novelty_operator, 
            start_fit_w=fit_w, 
            decay=decay,
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits 
        )
        toolbox.register("mutation", de.add_novelty_mutation)

class DiffSubNoveltyContainer(DiffAlgContainer, SubNovelty):
    """
    Implements differential evolution with additive mix of novelty and fitness
    """
    def __init__(
            self, 
            pop, 
            toolbox, 
            seed, 
            ngen, 
            creator, 
            fit_w, 
            decay,
            fitness_limits, 
            novelty_limits
        ):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        SubNovelty.__init__(self, ngen)
        toolbox.register(
            "map", 
            self.sub_novelty_operator, 
            start_fit_w=fit_w, 
            decay=decay,
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits 
        )
        toolbox.register(
            "dry_map", 
            self.dry_sub_novelty_operator, 
            start_fit_w=fit_w, 
            decay=decay,
            # fitness_limits = fitness_limits,
            # novelty_limits = novelty_limits 
        )
        toolbox.register("mutation", de.add_novelty_mutation)

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
    
class DiffArchivingEliteContainer(DiffAlgContainer):
    """
    Implements diffential evolution with basic archiving
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2, store_batch=1):
        super().__init__(pop, toolbox, seed, ngen, creator)
        self.archive = archiving.Archive(archiving_period, pop, store_batch=store_batch, selection=tools.selBest)

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

class LambdaArchivingEliteContainer(LambdaAlgContainer):
    """
    Implements L+M strategy with basic archiving
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2, store_batch=1):
        super().__init__(pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        self.archive = archiving.Archive(archiving_period, pop, store_batch=store_batch, selection=tools.selBest)
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


class LambdaArchivingLimitNoveltyContainer(LambdaArchivingContainer, PureNovelty):
    """
    Implements diffential evolution with basic archiving, pure novelty
    """
    def __init__(self, pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator, archiving_period=2, store_batch=1, archive_limit=0):
        LambdaArchivingContainer.__init__(self,pop, offs, mut_rate, cross_rate, seed, ngen, toolbox, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)        
        self.archive = archiving.LimitArchive(
            period=archiving_period, 
            size=pop, 
            store_batch=store_batch, 
            limit=archive_limit,
            fitness_attr=self.fit_attr
        )
        toolbox.decorate("select", self.archiveOp)



class DiffArchivingLimitNoveltyContainer(DiffAlgContainer, PureNovelty):
    """
    Implements diffential novelty evolution with basic archiving, pure novelty
    """
    def __init__(self, pop, toolbox, seed, ngen, creator, archiving_period=2, store_batch=1, archive_limit=0):
        DiffAlgContainer.__init__(self,pop, toolbox, seed, ngen, creator)
        PureNovelty.__init__(self)
        toolbox.register("map", self.novelty_operator)
        toolbox.register("mutation", de.novelty_mutation)
        self.archive = archiving.LimitArchive(
            period=archiving_period, 
            size=pop, 
            store_batch=store_batch,
            limit=archive_limit,
            fitness_attr=self.fit_attr
        )

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

