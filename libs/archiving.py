from deap.tools import selBest,selRandom
import numpy as np
import random
from operator import attrgetter
class Archive: #kouknout se do literatury na update archivu hlavně pro novelty!!!!!
    # novelty z poppulace a archivu
    # práce s archivem podle paperu
    def __init__(self, period, size, store_batch=1, selection=None):
        self.period = period
        self.size = size
        self.pop_storage = []
        if selection is None:
            self.selection = lambda x, n: random.sample(x,n)
        else: self.selection = selection
        self.batch = store_batch
        self.gn = 0

    def store_individuals(self, individual_list): #ukládat pouze ty významné
        self.gn = (self.gn + 1) % self.period
        if not self.gn:
            best_guys = self.selection(individual_list, self.batch)
            self.pop_storage += best_guys
            cutoff = max(0, len(self.pop_storage) - self.size)
            self.pop_storage =  self.pop_storage[cutoff:]#self.selection(self.pop_storage, self.size)

    def get_stored(self):
        if self.pop_storage == []: return []
        n = min(self.batch, len(self.pop_storage))
        return self.selection(self.pop_storage, n) #vyhodit jednoho nejelpšího


class LimitArchive(Archive):
    def __init__(self, period, size, limit=0, store_batch=1, fitness_attr="fitness2"):
        super().__init__(period, size, store_batch)
        self.limit = limit
        self.fitness_attr = fitness_attr
        self.getter = attrgetter(fitness_attr)

    def store_individuals(self, individual_list:list[object]): #ukládat pouze ty významné
        self.gn = (self.gn + 1) % self. period

        if self.gn % self. period:
            filtered = [ind for ind in individual_list if self.getter(ind).values[0] > self.limit]
            if filtered == []: return
            n = min(self.batch, len(filtered))
            best_guys = random.sample(filtered, n)
            self.pop_storage += best_guys
            cutoff = max(0, len(self.pop_storage) - self.size)
            self.pop_storage =  self.pop_storage[cutoff:]#self.selection(self.pop_storage, self.size)


class NoveltyArchive(Archive):
    def store_individuals(self, individual_list): #ukládat pouze ty významné
        self.gn = (self.gn + 1) % self. period

        if self.gn % self. period:
            newpop = individual_list + self.pop_storage
            behaviours = [ind.behaviour for ind in newpop]
            beh_distances = np.array(list(map(
                lambda x: np.linalg.norm(np.array(x[1])-np.array(behaviours), axis=1), 
                behaviours
            )))
            novelties = np.mean(beh_distances, axis=1)
            novlety_induced = list(zip(newpop, novelties))
            novlety_induced = sorted(novlety_induced, key=lambda x: x[1], reverse=True)
            best_guys = list(map(lambda x: x[0], novlety_induced[:self.size]))       
            self.pop_storage = best_guys

    def get_stored(self):
        return self.pop_storage[:self.batch]
# scatter search
