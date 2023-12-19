import numpy as np
from deap import tools
import typing

STAT_LIST = ["min","avg","max","std"]

STAT_DEF = {
    "avg": np.mean,
    "min": np.min,
    "max": np.max,
    "std": np.std
}

def get_statistics(f):
    stat_obj = tools.Statistics(f)
    for stat in STAT_DEF:
        stat_obj.register(stat, STAT_DEF[stat], axis=0)
    return stat_obj

def get_novelty_stats(novelty_f, fitness_f):
    fit_stats = tools.Statistics(fitness_f)
    novelty_stats = tools.Statistics(novelty_f)
    multi_stats = tools.MultiStatistics(fitness=fit_stats, novelty=novelty_stats)
    for stat in STAT_DEF:
        multi_stats.register(stat, STAT_DEF[stat], axis=0)
    return multi_stats
