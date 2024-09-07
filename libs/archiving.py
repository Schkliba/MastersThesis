from deap.tools import selBest
import numpy as np

class Archive: #kouknout se do literatury na update archivu hlavně pro novelty!!!!!
    # novelty z poppulace a archivu
    # práce s archivem podle paperu
    def __init__(self, period, size, store_batch=1):
        self.period = period
        self.size = size
        self.pop_storage = []
        self.selection = selBest
        self.batch = store_batch
        self.gn = 0

    def store_individuals(self, individual_list): #ukládat pouze ty významné
        self.gn = (self.gn + 1) % self. period
        if self.gn:
            best_guys = self.selection(individual_list, self.batch)
            self.pop_storage += best_guys
            cutoff = max(0, len(self.pop_storage) - self.size)
            self.pop_storage = self.selection(self.pop_storage, self.size)

    def get_stored(self):
        return self.selection(self.pop_storage, self.batch) #vyhodit jednoho nejelpšího


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
