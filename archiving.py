from deap.tools import selBest
class Archive: #kouknout se do literatury na update archivu hlavně pro novelty!!!!!
    # novelty z poppulace a archivu
    # práce s archivem podle paperu
    def __init__(self, period, size, store_batch=1):
        self.period = period
        self.size = size
        self.pop_storage = []
        self.selection = selBest
        self.batch = store_batch

    def store_individuals(self, gn, individual_list): #ukládat pouze ty významné
        if gn % self. period:
            best_guys = self.selection(individual_list, self.batch)
            self.pop_storage += best_guys
            cutoff = max(0, len(self.pop_storage) - self.size)
            self.pop_storage = self.selection(self.pop_storage, )

    def get_stored(self):
        return self.pop_storage #vyhodit jednoho nejelpšího


class NoveltyArchive(Archive):
    def store_individuals(self, gn, individual_list): #ukládat pouze ty významné
        if gn % self. period:
            newpop = individual_list + self.pop_storage
            behaviours = [ind.behaviour for ind in newpop]
            beh_distances = map(
                lambda x: np.linalg.norm(np.array(x[1])-np.array(behaviours)), 
                behaviours
            )
            novelties = np.mean(np.array(list(beh_distances)), axis=1)            
            self.pop_storage += best_guys
            cutoff = max(0, len(self.pop_storage) - self.size)
            self.pop_storage = self.pop_storage[cutoff:]
# scatter search
